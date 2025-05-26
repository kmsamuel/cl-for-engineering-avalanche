import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.benchmarks.utils import AvalancheDataset

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter
import random
import os
import sys
import hashlib
from avalanche.training.plugins import SupervisedPlugin

# Add BanditCL to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
banditcl_path = os.path.join(current_dir, "BanditCL")
sys.path.append(banditcl_path)

from NSMAB.Models.MovingAverage import MovingAverage
from NSMAB.Policies.Policies import NSMAB_Policy_Compositional
from NSMAB.Samplers import BoltzmannSampling

try:
    from tabpfn import TabPFNRegressor
except ImportError:
    TabPFNRegressor = None  # Only needed for the TabPFN variant

def get_model_param_hash(model: torch.nn.Module) -> str:
    """Returns an MD5 hash of the model's parameters."""
    with torch.no_grad():
        flat_params = torch.cat([p.data.view(-1).cpu() for p in model.parameters()])
    return hashlib.md5(flat_params.numpy().tobytes()).hexdigest()
def batch_hash(batch_x):
    return hashlib.md5(batch_x.cpu().numpy().tobytes()).hexdigest()
def get_optimizer_hash(optimizer):
    state_tensors = []
    for state in optimizer.state.values():
        for v in state.values():
            if isinstance(v, torch.Tensor):
                state_tensors.append(v.view(-1).cpu())
    if not state_tensors:
        return None
    all_data = torch.cat(state_tensors).numpy().tobytes()
    return hashlib.sha256(all_data).hexdigest()
def hash_model_params(model):
    param_bytes = b''.join([p.detach().cpu().numpy().tobytes() for p in model.parameters()])
    return hashlib.md5(param_bytes).hexdigest()

class BaseAdaptiveReplayPlugin(SupervisedPlugin):
    def __init__(self, mem_size=500, batch_size=32, batch_size_mem=5, temperature=1.0, n_clusters=4, vis_path=None):
        print(f"[DEBUG] BaseAdaptiveReplayPlugin __init__ called for {self.__class__.__name__}")
        super().__init__()
        print(f"[DEBUG] BaseAdaptiveReplayPlugin __init__ for {self.__class__.__name__} complete")


        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.temperature = temperature
        self.n_clusters = n_clusters
        self.cluster_forgetting_log = {}  # {cluster_id: [ [losses_exp0], [losses_exp1], ... ]}
        self.vis_path = vis_path

        self.buffer = []  # Stores (x, y, embedding, cluster_id, experience_id)

        modeller_ctor = lambda _: MovingAverage(alpha=0.7, assumed_increase_in_mean=10.0)
        sampling_strategy = BoltzmannSampling(softmax_temperature=self.temperature)
        self.bandit_policy = NSMAB_Policy_Compositional(modeller_ctor, sampling_strategy)
        self.bandit_policy.add_clusters(self.n_clusters)

        self.current_experience_id = 0
        
        self.training_iteration_count = 0
        self.log_every_n_iters = 1  

    def before_training_exp(self, strategy, **kwargs):
        self.selected_replay_embeddings = []
        self.selected_replay_ids = set()

        # Optional: diagnostics for experience 0 only
        if strategy.experience.current_experience == 0:
            print("[DEBUG] Param hash BEFORE training exp 0:", get_model_param_hash(strategy.model))
            print(f"[DEBUG] RNG state (first 5): {torch.get_rng_state()[:5]}")
            dataset_ids = [i for i in range(len(strategy.experience.dataset))]
            print("[DEBUG] Dataset sample indices (exp 0):", dataset_ids[:10])
            print("[DEBUG] Optimizer hash at start:", get_optimizer_hash(strategy.optimizer))


    def before_training_iteration(self, strategy, **kwargs):
        
        if self.current_experience_id == 0 or not self.buffer:
            # print(f"[DEBUG] Batch X hash at iter {strategy.clock.train_iterations}: {batch_hash(strategy.mb_x)}")
            # print("[DEBUG] Is buffer empty during exp 0? ", len(self.buffer) == 0)
            return

        device = strategy.device
        
        current_x = strategy.mb_x.cpu()
        current_y = strategy.mb_y.cpu()
        # print("[DEBUG] From plugin targets ", current_y)
        current_t = strategy.mb_task_id.cpu()
        # print("[DEBUG] current task ", current_t)

        # Step 1: Group buffer samples by cluster
        cluster_to_samples = {i: [] for i in range(self.n_clusters)}
        for x, y, emb, cid, eid in self.buffer:
            cluster_to_samples[cid].append((x, y, eid))

        # Step 2: Get cluster probabilities from bandit
        cluster_probs = self.bandit_policy.get_cluster_probabilities()

        # Step 3: Sample clusters
        selected_clusters = np.random.choice(
            np.arange(self.n_clusters),
            size=self.batch_size_mem,
            replace=True,
            p=cluster_probs
        )

        # Step 4: Sample memory points from selected clusters
        selected_replay_x, selected_replay_y, selected_replay_t = [], [], []

        for cid in selected_clusters:
            samples = cluster_to_samples.get(cid, [])
            if samples:
                x, y, eid = random.choice(samples)
                selected_replay_x.append(torch.tensor(x))
                selected_replay_y.append(torch.tensor(y))
                selected_replay_t.append(torch.tensor(eid))

                # Track unique sample replayed
                sample_id = (eid, tuple(x))
                if sample_id not in self.selected_replay_ids:
                    self.selected_replay_ids.add(sample_id)
                    for entry in self.buffer:
                        if np.allclose(entry[0], x) and entry[1] == y and entry[4] == eid:
                            self.selected_replay_embeddings.append(entry[2])
                            break

        # Step 5: Merge current batch with memory samples
        if selected_replay_x:
            selected_replay_x = torch.stack(selected_replay_x).to(device)
            selected_replay_y = torch.stack(selected_replay_y).to(device)
            selected_replay_t = torch.stack(selected_replay_t).to(device)

            strategy.mbatch = (
                torch.cat([current_x.to(device), selected_replay_x], dim=0),
                torch.cat([current_y.to(device), selected_replay_y], dim=0),
                torch.cat([current_t.to(device), selected_replay_t], dim=0)
            )
        
        ## Adding Logging of Clustering
        if self.training_iteration_count % self.log_every_n_iters == 0:
            cluster_logs = self._log_losses_to_bandit(strategy)
            for cid, loss_list in cluster_logs.items():
                if cid not in self.cluster_forgetting_log:
                    self.cluster_forgetting_log[cid] = []
                self.cluster_forgetting_log[cid].append(loss_list)

        self.training_iteration_count += 1

    def after_training_exp(self, strategy, **kwargs):
        # Add current experience to buffer
        print(f"[DEBUG] after_training_exp called for {self.__class__.__name__}")
        print(f"[DEBUG] Checking if _extract_data_from_dataset exists: {hasattr(self, '_extract_data_from_dataset')}")
        print(f"[DEBUG] About to call _extract_data_from_dataset for {self.__class__.__name__}")
        
        X, y = self._extract_data_from_dataset(strategy.experience.dataset)
        current_exp_id = strategy.experience.current_experience
        X_new = np.stack(X)
        y_new = np.stack(y)

        embeddings = self.compute_embeddings(X_new, y_new)
        new_entries = [(x, y_val, emb, -1, current_exp_id)
                    for x, y_val, emb in zip(X_new, y_new, embeddings)]
        self.buffer.extend(new_entries)
        assert all(len(entry) == 5 for entry in self.buffer), "Buffer entry structure invalid!"

        # Reassign clusters
        all_embeddings = np.stack([entry[2] for entry in self.buffer])
        cluster_ids = self.assign_clusters(all_embeddings)
        self.buffer = [(x, y, emb, cid, eid) for (cid, (x, y, emb, _, eid)) in zip(cluster_ids, self.buffer)]
        
        exp_ids = sorted(set(eid for (_, _, _, _, eid) in self.buffer))
        samples_per_exp = max(1, self.mem_size // len(exp_ids))
        new_buffer = []
        for eid in exp_ids:
            entries = [e for e in self.buffer if e[4] == eid]
            new_buffer.extend(random.sample(entries, min(samples_per_exp, len(entries))))
        self.buffer = new_buffer
        print(f"[DEBUG] Buffer rebalanced and truncated to {len(self.buffer)}")
        
        self.visualize_clusters(
            save_path=os.path.join(self.vis_path, f"experience_{self.current_experience_id}_clusters.png"),
            title=f"Clusters After Experience {self.current_experience_id}"
        )
        self.visualize_selected_samples(
            save_path=os.path.join(self.vis_path, f"experience_{self.current_experience_id}_selected_samples.png"),
            title=f"Replay Samples After Experience {self.current_experience_id}"
        )

        # Reset bandit policy after reclustering
        modeller_ctor = lambda _: MovingAverage(alpha=0.7, assumed_increase_in_mean=10.0)
        sampling_strategy = BoltzmannSampling(softmax_temperature=self.temperature)
        self.bandit_policy = NSMAB_Policy_Compositional(modeller_ctor, sampling_strategy)
        self.bandit_policy.add_clusters(self.n_clusters)
        print(f"[DEBUG] Bandit policy reset after reclustering with {self.n_clusters} new clusters.")

        # Update bandit model based on fresh computed losses
        X_buf = np.stack([x for (x, _, _, _, _) in self.buffer])
        y_buf = np.stack([y for (_, y, _, _, _) in self.buffer])
        cid_buf = [cid for (_, _, _, cid, _) in self.buffer]

        with torch.no_grad():
            model = strategy.model.to(strategy.device)
            model.eval()
            preds = model(torch.tensor(X_buf).to(strategy.device))
            losses = torch.nn.functional.mse_loss(preds.squeeze(), torch.tensor(y_buf.squeeze()).to(strategy.device), reduction='none')
            losses = losses.cpu().numpy()

        if strategy.model.training == False:
            model.train() 
            
        cluster_logs = {}
        for cid, loss in zip(cid_buf, losses):
            cluster_logs.setdefault(cid, []).append(loss)

        self.bandit_policy.log_observations({cid: np.array(l) for cid, l in cluster_logs.items()})
        self.bandit_policy.next_timestep()

        # Track per-cluster loss over time (for forgetting visualization)
        for cid, loss_list in cluster_logs.items():
            if cid not in self.cluster_forgetting_log:
                self.cluster_forgetting_log[cid] = []
            self.cluster_forgetting_log[cid].append(loss_list)  # list of losses for this cluster at this experience
            
        self.current_experience_id += 1

        print(f"[AdaptiveReplay] Buffer updated: {len(self.buffer)} samples across {self.n_clusters} clusters.")
        print(f"[AdaptiveReplay] Total unique replay embeddings selected this experience: {len(self.selected_replay_embeddings)}")

    def before_eval(self, strategy, **kwargs):
        print("[DEBUG] Model eval mode:", strategy.model.training == False)
        
    def after_eval(self, strategy, **kwargs):
        print("[DEBUG] Inside after_eval hook")
        print("[DEBUG] Param hash AFTER eval exp :", get_model_param_hash(strategy.model))
        print("[DEBUG] Param hash state dict AFTER training exp and clustering:", hash_model_params(strategy.model))        
        # Print out first few predictions and targets from the last batch
        if hasattr(strategy, 'mb_output') and hasattr(strategy, 'mb_y'):
            preds = strategy.mb_output.detach().cpu()
            targets = strategy.mb_y.detach().cpu()
            print(f"[DEBUG] Eval preds (first 5): {preds[:5]}")
            print(f"[DEBUG] Eval targets (first 5): {targets[:5]}")
        else:
            print("[DEBUG] Eval outputs not available in strategy.")
        
    def compute_embeddings(self, X, y):
        # Default: no embeddings, just return inputs
        return X

    def assign_clusters(self, embeddings):
        raise NotImplementedError("assign_clusters must be implemented in subclasses.")

    def _extract_data_from_dataset(self, dataset):
        X, y = [], []
        for i in range(len(dataset)):
            sample = dataset[i]
            if isinstance(sample, tuple) and len(sample) >= 2:
                X.append(sample[0].cpu().numpy() if isinstance(sample[0], torch.Tensor) else sample[0])
                y.append(sample[1])
        return np.array(X), np.array(y)
    
    def _log_losses_to_bandit(self, strategy):
        X_buf = np.stack([x for (x, _, _, _, _) in self.buffer])
        y_buf = np.stack([y for (_, y, _, _, _) in self.buffer])
        cid_buf = [cid for (_, _, _, cid, _) in self.buffer]

        with torch.no_grad():
            model = strategy.model.to(strategy.device)
            model.eval()
            preds = model(torch.tensor(X_buf).to(strategy.device))
            losses = torch.nn.functional.mse_loss(preds.squeeze(), torch.tensor(y_buf.squeeze()).to(strategy.device), reduction='none')
            losses = losses.cpu().numpy()

        if strategy.model.training == False:
            model.train() 

        cluster_logs = {}
        for cid, loss in zip(cid_buf, losses):
            cluster_logs.setdefault(cid, []).append(loss)

        self.bandit_policy.log_observations({cid: np.array(l) for cid, l in cluster_logs.items()})
        self.bandit_policy.next_timestep()
        
        return cluster_logs
        
    def visualize_clusters(self, save_path=None, title="Buffer Clusters Visualization"):
        """ Visualizes the buffer clustering using PCA projection. """
        if not self.buffer:
            print("[DEBUG] Buffer is empty. Nothing to visualize.")
            return
        
        # Extract embeddings and cluster IDs
        embeddings = np.stack([entry[2] for entry in self.buffer])
        cluster_ids = np.array([entry[3] for entry in self.buffer])

        # Project to 2D using PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        # Plot
        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1], 
            c=cluster_ids, cmap='tab10', alpha=0.7, edgecolors='k'
        )
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(title)
        plt.xlabel('PCA Dimension 1')
        plt.ylabel('PCA Dimension 2')
        
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"[DEBUG] Cluster visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
            
    def visualize_selected_samples(self, save_path=None, title="Selected Replay Samples"):
        # if not self.buffer or not hasattr(self, 'selected_replay_embeddings'):
        #     print("[DEBUG] No selected replay samples to visualize.")
        #     return
        if not self.buffer or not hasattr(self, 'selected_replay_embeddings') or len(self.selected_replay_embeddings) == 0:
            print("[DEBUG] No selected replay samples to visualize.")
            return

        # all_embeddings = np.stack([entry[2] for entry in self.buffer])
        # selected_embeddings = np.stack(self.selected_replay_embeddings)
        all_embeddings = np.vstack([entry[2] for entry in self.buffer])

        selected_embeddings = np.vstack(self.selected_replay_embeddings)
        
        # Ensure both are 2D
        if len(all_embeddings.shape) == 1:
            all_embeddings = all_embeddings.reshape(1, -1)
        if len(selected_embeddings.shape) == 1:
            selected_embeddings = selected_embeddings.reshape(1, -1)

        pca = PCA(n_components=2)
        all_2d = pca.fit_transform(all_embeddings)
        selected_2d = pca.transform(selected_embeddings)

        plt.figure(figsize=(8, 8))
        plt.scatter(all_2d[:, 0], all_2d[:, 1], c="#cccccc", alpha=0.5, label="All Buffer Samples")
        plt.scatter(selected_2d[:, 0], selected_2d[:, 1], c="red", label="Selected Replay Samples")

        plt.title(title)
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"[DEBUG] Replay visualization saved to {save_path}")
        else:
            plt.show()
        plt.close()
        
    def plot_forgetting_dynamics(self, save_path=None):
        if not hasattr(self, 'cluster_forgetting_log') or not self.cluster_forgetting_log:
            print("[DEBUG] No forgetting data to plot.")
            return

        plt.figure(figsize=(10, 6))
        for cid, losses_over_time in self.cluster_forgetting_log.items():
            means = [np.mean(x) for x in losses_over_time]
            stds = [np.std(x) for x in losses_over_time]
            x = np.arange(len(means))
            plt.plot(x, means, label=f"Cluster {cid}")
            plt.fill_between(x, np.array(means) - stds, np.array(means) + stds, alpha=0.3)

        plt.title("True Forgetting Distribution Per Cluster Over Time")
        plt.xlabel("Training Interval (Experience Index)")
        plt.ylabel("Loss (used as forgetting proxy)")
        plt.legend()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"[DEBUG] Forgetting dynamics plot saved to: {save_path}")
        else:
            plt.show()
        plt.close()

    def save_forgetting_log_csv(self, csv_path):
        """
        Save the forgetting log as a CSV with rows:
        [experience_id, cluster_id, mean_loss, std_loss]
        """
        if not self.cluster_forgetting_log:
            print("[DEBUG] No forgetting data to save.")
            return

        rows = []
        for cid, losses_over_time in self.cluster_forgetting_log.items():
            for exp_id, loss_list in enumerate(losses_over_time):
                mean_loss = np.mean(loss_list)
                std_loss = np.std(loss_list)
                rows.append({
                    "experience_id": exp_id,
                    "cluster_id": cid,
                    "mean_loss": mean_loss,
                    "std_loss": std_loss
                })

        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"[DEBUG] Forgetting log saved to: {csv_path}")

class ExperienceAdaptiveReplayPlugin(BaseAdaptiveReplayPlugin):
    def __init__(self, mem_size=500, batch_size=32, batch_size_mem=5, temperature=1.0, vis_path=None):
        print(f"[DEBUG] {self.__class__.__name__} initializing...")
        super().__init__(
            mem_size=mem_size,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            temperature=temperature,
            n_clusters=1,  # Start with 1 cluster at exp 0
            vis_path=vis_path
        )
        print(f"[DEBUG] {self.__class__.__name__} super().__init__ complete.")

    def assign_clusters(self, embeddings):
        # Cluster assignment = experience ID directly
        return np.array([eid for (_, _, _, _, eid) in self.buffer])

    def after_training_exp(self, strategy, **kwargs):
        print(f"[DEBUG] after_training_exp called for {self.__class__.__name__}")

        # Step 1: Extract and add current experience to buffer
        X, y = self._extract_data_from_dataset(strategy.experience.dataset)
        current_exp_id = strategy.experience.current_experience
        X_new = np.stack(X)
        y_new = np.stack(y)

        embeddings = self.compute_embeddings(X_new, y_new)
        new_entries = [(x, y_val, emb, -1, current_exp_id)
                       for x, y_val, emb in zip(X_new, y_new, embeddings)]
        self.buffer.extend(new_entries)
        assert all(len(entry) == 5 for entry in self.buffer), "Buffer entry structure invalid!"

        # Step 2: Update number of clusters if needed
        if current_exp_id >= self.n_clusters:
            print(f"[DEBUG] Adding new cluster for experience {current_exp_id}")
            self.bandit_policy.add_clusters(1)
            self.n_clusters += 1

        # Step 3: Reassign clusters
        all_embeddings = np.stack([entry[2] for entry in self.buffer])
        cluster_ids = self.assign_clusters(all_embeddings)
        self.buffer = [(x, y, emb, cid, eid) for (cid, (x, y, emb, _, eid)) in zip(cluster_ids, self.buffer)]

        # Step 4: Visualize clusters
        self.visualize_clusters(
            save_path=os.path.join(self.vis_path, f"experience_{self.current_experience_id}_clusters.png"),
            title=f"Clusters After Experience {self.current_experience_id}"
        )
        self.visualize_selected_samples(
            save_path=os.path.join(self.vis_path, f"experience_{self.current_experience_id}_selected_samples.png"),
            title=f"Replay Samples After Experience {self.current_experience_id}"
        )
        
        # Step 5: Update bandit with loss info
        X_buf = np.stack([x for (x, _, _, _, _) in self.buffer])
        y_buf = np.stack([y for (_, y, _, _, _) in self.buffer])
        cid_buf = [cid for (_, _, _, cid, _) in self.buffer]

        with torch.no_grad():
            model = strategy.model.to(strategy.device)
            preds = model(torch.tensor(X_buf).to(strategy.device))
            losses = torch.nn.functional.mse_loss(preds.squeeze(), torch.tensor(y_buf.squeeze()).to(strategy.device), reduction='none')
            losses = losses.cpu().numpy()

        cluster_logs = {}
        for cid, loss in zip(cid_buf, losses):
            cluster_logs.setdefault(cid, []).append(loss)

        self.bandit_policy.log_observations({cid: np.array(l) for cid, l in cluster_logs.items()})
        self.bandit_policy.next_timestep()
        
        # Track per-cluster loss over time (for forgetting visualization)
        for cid, loss_list in cluster_logs.items():
            if cid not in self.cluster_forgetting_log:
                self.cluster_forgetting_log[cid] = []
            self.cluster_forgetting_log[cid].append(loss_list)  # list of losses for this cluster at this experience
        
        total_experiences = len(strategy.experience.origin_stream)

        if self.current_experience_id + 1 == total_experiences:
            save_path = os.path.join(self.vis_path, "final_forgetting_dynamics.png")
            self.plot_forgetting_dynamics(save_path=save_path)

            csv_path = os.path.join(self.vis_path, "final_forgetting_dynamics.csv")
            self.save_forgetting_log_csv(csv_path)
                
        param_hash = get_model_param_hash(strategy.model)
        print(f"[DEBUG] Experience {strategy.experience.current_experience} param hash: {param_hash}")

        self.current_experience_id += 1
        print(f"[AdaptiveReplay] Buffer updated: {len(self.buffer)} samples across {self.n_clusters} clusters.")
        print(f"[AdaptiveReplay] Total unique replay embeddings selected this experience: {len(self.selected_replay_embeddings)}")


# class KMeansInputAdaptiveReplayPlugin(BaseAdaptiveReplayPlugin):
#     def __init__(self, mem_size=500, batch_size=32, temperature=1.0, n_clusters=4, vis_path=None):
#         print(f"[DEBUG] {self.__class__.__name__} initializing...")

#         super().__init__(
#             mem_size=mem_size,
#             batch_size=batch_size,
#             temperature=temperature,
#             n_clusters=n_clusters,
#             vis_path=vis_path
#         )
#         print(f"[DEBUG] {self.__class__.__name__} super().__init__ complete.")


#     def assign_clusters(self, embeddings):
#         kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
#         return kmeans.fit_predict(embeddings)

class KMeansInputAdaptiveReplayPlugin(BaseAdaptiveReplayPlugin):
    def __init__(self, mem_size=500, batch_size=32, batch_size_mem=5, temperature=1.0, vis_path=None):
        print(f"[DEBUG] {self.__class__.__name__} initializing...")
        super().__init__(
            mem_size=mem_size,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            temperature=temperature,
            n_clusters=1,  # Start with 1 cluster at exp 0
            vis_path=vis_path
        )
        print(f"[DEBUG] {self.__class__.__name__} super().__init__ complete.")

    def assign_clusters(self, embeddings):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        return kmeans.fit_predict(embeddings)

    def after_training_exp(self, strategy, **kwargs):
        print(f"[DEBUG] after_training_exp called for {self.__class__.__name__}")

        # Step 1: Add current experience to buffer
        X, y = self._extract_data_from_dataset(strategy.experience.dataset)
        current_exp_id = strategy.experience.current_experience
        X_new = np.stack(X)
        y_new = np.stack(y)
        embeddings = self.compute_embeddings(X_new, y_new)
        new_entries = [(x, y_val, emb, -1, current_exp_id)
                       for x, y_val, emb in zip(X_new, y_new, embeddings)]
        self.buffer.extend(new_entries)

        # Step 2: Update number of clusters
        self.n_clusters = current_exp_id + 1

        # Step 3: Reassign clusters using KMeans over inputs
        all_embeddings = np.stack([entry[2] for entry in self.buffer])
        cluster_ids = self.assign_clusters(all_embeddings)
        self.buffer = [(x, y, emb, cid, eid) for (cid, (x, y, emb, _, eid)) in zip(cluster_ids, self.buffer)]
        
        # Step 3.5: Visualize clusters
        self.visualize_clusters(
            save_path=os.path.join(self.vis_path, f"experience_{self.current_experience_id}_clusters.png"),
            title=f"Clusters After Experience {self.current_experience_id}"
        )
        self.visualize_selected_samples(
            save_path=os.path.join(self.vis_path, f"experience_{self.current_experience_id}_selected_samples.png"),
            title=f"Replay Samples After Experience {self.current_experience_id}"
        )
        
        # Step 4: Reset and reinitialize bandit
        modeller_ctor = lambda _: MovingAverage(alpha=0.7, assumed_increase_in_mean=10.0)
        sampling_strategy = BoltzmannSampling(softmax_temperature=self.temperature)
        self.bandit_policy = NSMAB_Policy_Compositional(modeller_ctor, sampling_strategy)
        self.bandit_policy.add_clusters(self.n_clusters)

        # Step 5: Log losses to bandit
        cluster_logs = self._log_losses_to_bandit(strategy)

        # Track per-cluster loss over time (for forgetting visualization)
        for cid, loss_list in cluster_logs.items():
            if cid not in self.cluster_forgetting_log:
                self.cluster_forgetting_log[cid] = []
            self.cluster_forgetting_log[cid].append(loss_list)  # list of losses for this cluster at this experience
        
        total_experiences = len(strategy.experience.origin_stream)
        
        if self.current_experience_id + 1 == total_experiences:
            save_path = os.path.join(self.vis_path, "final_forgetting_dynamics.png")
            self.plot_forgetting_dynamics(save_path=save_path)

            csv_path = os.path.join(self.vis_path, "final_forgetting_dynamics.csv")
            self.save_forgetting_log_csv(csv_path)
                        
        self.current_experience_id += 1



# class TabPFNAdaptiveReplayPlugin(BaseAdaptiveReplayPlugin):
#     def __init__(self, mem_size=500, batch_size=32, temperature=1.0, n_clusters=4, vis_path=None):
#         print(f"[DEBUG] {self.__class__.__name__} initializing...")

#         if TabPFNRegressor is None:
#             raise ImportError("TabPFN must be installed to use TabPFNAdaptiveReplayPlugin!")
#         super().__init__(
#             mem_size=mem_size,
#             batch_size=batch_size,
#             temperature=temperature,
#             n_clusters=n_clusters,
#             vis_path=vis_path
#         )
#         print(f"[DEBUG] {self.__class__.__name__} super().__init__ complete.")

#         self.tabpfn_model = TabPFNRegressor()

#     def compute_embeddings(self, X, y):
#         X_tensor = torch.tensor(X.astype(np.float32))
#         y_tensor = torch.tensor(y.astype(np.float32).ravel())
#         try:
#             self.tabpfn_model.fit(X_tensor, y_tensor)
#             self.tabpfn_model.predict(X_tensor, output_type='mean')
#             embeddings = self.tabpfn_model.model_.train_encoder_out.squeeze(1)
#             return embeddings.detach().cpu().numpy()
#         except Exception as e:
#             print(f"TabPFN embedding extraction failed: {e}")
#             return X  # fallback to inputs

#     def assign_clusters(self, embeddings):
#         kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
#         return kmeans.fit_predict(embeddings)

class TabPFNAdaptiveReplayPlugin(BaseAdaptiveReplayPlugin):
    def __init__(self, mem_size=500, batch_size=32, batch_size_mem=5, temperature=1.0, vis_path=None):
        print(f"[DEBUG] {self.__class__.__name__} initializing...")

        if TabPFNRegressor is None:
            raise ImportError("TabPFN must be installed to use TabPFNAdaptiveReplayPlugin!")

        super().__init__(
            mem_size=mem_size,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            temperature=temperature,
            n_clusters=1,  # Start with 1 cluster at exp 0
            vis_path=vis_path
        )
        
        self.tabpfn_model = TabPFNRegressor()
        print(f"[DEBUG] {self.__class__.__name__} super().__init__ complete.")

    def compute_embeddings(self, X, y):
        X_tensor = torch.tensor(X.astype(np.float32))
        y_tensor = torch.tensor(y.astype(np.float32).ravel())
        try:
            self.tabpfn_model.fit(X_tensor, y_tensor)
            self.tabpfn_model.predict(X_tensor, output_type='mean')
            embeddings = self.tabpfn_model.model_.train_encoder_out.squeeze(1)
            return embeddings.detach().cpu().numpy()
        except Exception as e:
            print(f"TabPFN embedding extraction failed: {e}")
            return X  # fallback to inputs

    def assign_clusters(self, embeddings):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        return kmeans.fit_predict(embeddings)

    def after_training_exp(self, strategy, **kwargs):
        print(f"[DEBUG] after_training_exp called for {self.__class__.__name__}")
        
        if self.current_experience_id == 0:
            print("[DEBUG] Param hash AFTER training exp 0:", get_model_param_hash(strategy.model))
        # Step 1: Add current experience to buffer
        X, y = self._extract_data_from_dataset(strategy.experience.dataset)
        current_exp_id = strategy.experience.current_experience
        X_new = np.stack(X)
        y_new = np.stack(y)
        embeddings = self.compute_embeddings(X_new, y_new)
        new_entries = [(x, y_val, emb, -1, current_exp_id)
                       for x, y_val, emb in zip(X_new, y_new, embeddings)]
        self.buffer.extend(new_entries)

        # Step 2: Update number of clusters
        self.n_clusters = current_exp_id + 1

        # Step 3: Reassign clusters using TabPFN embeddings
        all_embeddings = np.stack([entry[2] for entry in self.buffer])
        cluster_ids = self.assign_clusters(all_embeddings)
        self.buffer = [(x, y, emb, cid, eid) for (cid, (x, y, emb, _, eid)) in zip(cluster_ids, self.buffer)]
        
        # Step 3.5: Visualize clusters
        self.visualize_clusters(
            save_path=os.path.join(self.vis_path, f"experience_{self.current_experience_id}_clusters.png"),
            title=f"Clusters After Experience {self.current_experience_id}"
        )
        self.visualize_selected_samples(
            save_path=os.path.join(self.vis_path, f"experience_{self.current_experience_id}_selected_samples.png"),
            title=f"Replay Samples After Experience {self.current_experience_id}"
        )
        # Step 4: Reset and reinitialize bandit
        modeller_ctor = lambda _: MovingAverage(alpha=0.7, assumed_increase_in_mean=10.0)
        sampling_strategy = BoltzmannSampling(softmax_temperature=self.temperature)
        self.bandit_policy = NSMAB_Policy_Compositional(modeller_ctor, sampling_strategy)
        self.bandit_policy.add_clusters(self.n_clusters)

        # Step 5: Log losses to bandit
        cluster_logs = self._log_losses_to_bandit(strategy)

        # Track per-cluster loss over time (for forgetting visualization)
        for cid, loss_list in cluster_logs.items():
            if cid not in self.cluster_forgetting_log:
                self.cluster_forgetting_log[cid] = []
            self.cluster_forgetting_log[cid].append(loss_list)  # list of losses for this cluster at this experience
        total_experiences = len(strategy.experience.origin_stream)
                
        if self.current_experience_id + 1 == total_experiences:
            save_path = os.path.join(self.vis_path, "final_forgetting_dynamics.png")
            self.plot_forgetting_dynamics(save_path=save_path)

            csv_path = os.path.join(self.vis_path, "final_forgetting_dynamics.csv")
            self.save_forgetting_log_csv(csv_path)

        if self.current_experience_id == 0:
            print("[DEBUG] Param hash AFTER training exp 0 and clustering:", get_model_param_hash(strategy.model))
            print("[DEBUG] Param hash state dict AFTER training exp 0 and clustering:", hash_model_params(strategy.model))

        self.current_experience_id += 1



__all__ = [
    "BaseAdaptiveReplayPlugin",
    "ExperienceAdaptiveReplayPlugin",
    "KMeansInputAdaptiveReplayPlugin",
    "TabPFNAdaptiveReplayPlugin"
]

if __name__ == "__main__":
    print("Adaptive Replay Plugins Loaded.")
