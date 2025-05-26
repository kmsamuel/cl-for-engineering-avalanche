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
sys.path.append(current_dir)

from enhanced_visualization import EnhancedVisualizationTracker
from NSMAB.Models.MovingAverage import MovingAverage
from NSMAB.Policies.Policies import NSMAB_Policy_Compositional
from NSMAB.Samplers import BoltzmannSampling

try:
    from tabpfn import TabPFNRegressor, TabPFNClassifier
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
        self.vis_path = vis_path

        self.buffer = []  # Stores (x, y, embedding, cluster_id, experience_id)
        self.selected_replay_embeddings = []
        self.selected_replay_ids = set()
        
        
        # Initialize bandit policy
        self._init_bandit_policy()
        
        # Tracking structures
        self.cluster_forgetting_log = {}  # {cluster_id: [ [losses_exp0], [losses_exp1], ... ]}
        self.sample_best_loss = {}  # {(eid, tuple(x)): best_loss}
        self.current_experience_id = 0
        # Training iteration tracking
        self.training_iteration = 0  # Global iteration counter across all experiences
        self.current_exp_iterations = 0  # Iteration counter within current experience
        self.forgetting_tracking_interval = 10  # How often to track forgetting (every N iterations)
        self.iteration_forgetting_log = []  # [{iteration, exp_id, cluster_id, mean_forgetting, std_forgetting}]
        
        # Visualization tracker
        self.viz_tracker = EnhancedVisualizationTracker(base_dir=self.vis_path)
    
    def _init_bandit_policy(self):
        """Initialize or reset the bandit policy"""
        modeller_ctor = lambda _: MovingAverage(alpha=0.7, assumed_increase_in_mean=10.0)
        sampling_strategy = BoltzmannSampling(softmax_temperature=self.temperature)
        self.bandit_policy = NSMAB_Policy_Compositional(modeller_ctor, sampling_strategy)
        self.bandit_policy.add_clusters(self.n_clusters)
        print(f"[DEBUG] Bandit policy initialized with {self.n_clusters} clusters")

    def before_training_exp(self, strategy, **kwargs):
        """Called before training on each experience"""
        self.selected_replay_embeddings = []
        self.selected_replay_ids = set()

    def before_training_iteration(self, strategy, **kwargs):
        """Before each training iteration/batch, add replay samples if needed"""
        # Skip if we're on the first experience or buffer is empty
        if self.current_experience_id == 0 or not self.buffer:
            return

        device = strategy.device
        
        current_x = strategy.mb_x.cpu()
        current_y = strategy.mb_y.cpu()
        current_t = strategy.mb_task_id.cpu()
        
        # Update bandit with latest loss information for each cluster
        # This implements steps 6-10 from the pseudocode
        cluster_logs = self._compute_losses_per_cluster(strategy)
        
        # Track per-cluster loss for visualization
        for cid, loss_list in cluster_logs.items():
            if cid not in self.cluster_forgetting_log:
                self.cluster_forgetting_log[cid] = []
                
            # Make sure we have enough entries in the forgetting log
            while len(self.cluster_forgetting_log[cid]) <= self.current_experience_id:
                self.cluster_forgetting_log[cid].append([])
                
            self.cluster_forgetting_log[cid][self.current_experience_id] = loss_list
        
        # Update visualization tracker with forgetting
        self.viz_tracker.update_forgetting(self.current_experience_id, cluster_logs)
        
        # Step 1: Group buffer samples by cluster
        cluster_to_samples = {i: [] for i in range(self.n_clusters)}
        for x, y, emb, cid, eid in self.buffer:
            cluster_to_samples[cid].append((x, y, eid))

        # Step 2: Get cluster probabilities from bandit
        cluster_probs = self.bandit_policy.get_cluster_probabilities()

        # Track cluster probabilities in visualization
        self.viz_tracker.update_cluster_probabilities(self.current_experience_id, cluster_probs)
        
        # Track cluster sizes
        cluster_sizes = {cid: len(samples) for cid, samples in cluster_to_samples.items()}
        self.viz_tracker.update_cluster_sizes(self.current_experience_id, cluster_sizes)
        
        # Step 3: Sample clusters based on probabilities (step 11 in pseudocode)
        selected_clusters = np.random.choice(
            np.arange(self.n_clusters),
            size=self.batch_size_mem,
            replace=True,
            p=cluster_probs
        )
        # Track which clusters were selected
        self.viz_tracker.update_cluster_selections(self.current_experience_id, selected_clusters)

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

        # Step 5: Merge current batch with memory samples (adding to it, not replacing)
        if selected_replay_x:
            selected_replay_x = torch.stack(selected_replay_x).to(device)
            selected_replay_y = torch.stack(selected_replay_y).to(device)
            selected_replay_t = torch.stack(selected_replay_t).to(device)
            
            strategy.mbatch = (
                torch.cat([current_x.to(device), selected_replay_x], dim=0),
                torch.cat([current_y.to(device), selected_replay_y], dim=0),
                torch.cat([current_t.to(device), selected_replay_t], dim=0)
            )
            
        # Track iteration-level forgetting data
        if self.training_iteration % self.forgetting_tracking_interval == 0:
            self._track_iteration_forgetting(cluster_logs)
            
            # Save iteration-level forgetting data after some intervals
            if self.training_iteration % (10 * self.forgetting_tracking_interval) == 0:
                self._save_iteration_forgetting_log()
            
        # Advance the bandit policy timestep
        self.bandit_policy.next_timestep()
        # Update iteration counters
        self.training_iteration += 1
        self.current_exp_iterations += 1
        
    def after_training_exp(self, strategy, **kwargs):
        """After training on each experience, update buffer and bandit"""
        # Extract data from current experience
        X, y = self._extract_data_from_dataset(strategy.experience.dataset)
        current_exp_id = strategy.experience.current_experience
        X_new = np.stack(X)
        y_new = np.stack(y)

        # Compute embeddings and add to buffer
        embeddings = self.compute_embeddings(X_new, y_new)
        new_entries = [(x, y_val, emb, -1, current_exp_id)
                    for x, y_val, emb in zip(X_new, y_new, embeddings)]
        self.buffer.extend(new_entries)
        
        # Reassign clusters for all samples in buffer
        all_embeddings = np.stack([entry[2] for entry in self.buffer])
        cluster_ids = self.assign_clusters(all_embeddings)
        self.buffer = [(x, y, emb, cid, eid) for (cid, (x, y, emb, _, eid)) in zip(cluster_ids, self.buffer)]
        
        # Maintain buffer size by sampling from each experience
        self._balance_buffer()
        
        # Reset the bandit policy after clustering
        # This ensures a fresh start for the next experience with the new clusters
        self._init_bandit_policy()
        
        # Generate visualizations
        self._generate_visualizations(strategy)
        
        # Save iteration-level forgetting data
        self._save_iteration_forgetting_log()

        # Reset per-experience iteration counter
        self.current_exp_iterations = 0

        # Update experience counter and report
        self.current_experience_id += 1
        print(f"[AdaptiveReplay] Buffer updated: {len(self.buffer)} samples across {self.n_clusters} clusters.")
        print(f"[AdaptiveReplay] Total unique replay embeddings selected this experience: {len(self.selected_replay_embeddings)}")

    def _balance_buffer(self):
        """Balance buffer to maintain max size and even distribution across experiences"""
        exp_ids = sorted(set(eid for (_, _, _, _, eid) in self.buffer))
        samples_per_exp = max(1, self.mem_size // len(exp_ids))
        
        new_buffer = []
        for eid in exp_ids:
            entries = [e for e in self.buffer if e[4] == eid]
            if entries:
                new_buffer.extend(random.sample(entries, min(samples_per_exp, len(entries))))
        
        self.buffer = new_buffer
        print(f"[DEBUG] Buffer rebalanced and truncated to {len(self.buffer)}")

    def _compute_losses_per_cluster(self, strategy):
        """
        Compute forgetting for all samples in buffer to update bandit policy
        This implements steps 6-9 from the bandit pseudocode:
        6. For each cluster Ai do
        7. Sample points from Ai to est. forgetting
        8. Update means with exponential moving average
        9. End for
        """
        if not self.buffer:
            return {}
            
        X_buf = np.stack([x for (x, _, _, _, _) in self.buffer])
        y_buf = np.stack([y for (_, y, _, _, _) in self.buffer])
        cid_buf = [cid for (_, _, _, cid, _) in self.buffer]
        eid_buf = [eid for (_, _, _, _, eid) in self.buffer]

        # Compute current model's performance on buffer samples
        with torch.no_grad():
            model = strategy.model.to(strategy.device)
            was_training = model.training
            model.eval()
            preds = model(torch.tensor(X_buf).to(strategy.device))
            losses = torch.nn.functional.mse_loss(preds.squeeze(), torch.tensor(y_buf.squeeze()).to(strategy.device), reduction='none')
            losses = losses.cpu().numpy()

        # Restore model's training state if needed
        if was_training:
            model.train()
            
        # Calculate forgetting for each sample and track by cluster
        cluster_logs = {}
        
        for i, ((x_id, y_id), loss_val, cid, eid) in enumerate(zip(zip(map(tuple, X_buf), y_buf), losses, cid_buf, eid_buf)):
            # Only compute forgetting for samples from previous experiences
            # (skip samples from the current experience)
            if eid >= self.current_experience_id:
                continue
                
            sample_id = (eid, tuple(x_id))
            
            # If this is the first time we've seen this sample, record its initial loss
            if sample_id not in self.sample_best_loss:
                self.sample_best_loss[sample_id] = loss_val
                forgetting = 0.0  # No forgetting yet
            else:
                # Calculate forgetting as difference between current loss and best previous loss
                baseline_loss = self.sample_best_loss[sample_id]
                forgetting = max(0, loss_val - baseline_loss)  # Only positive forgetting
                
                # Update best loss if current performance is better
                if loss_val < self.sample_best_loss[sample_id]:
                    self.sample_best_loss[sample_id] = loss_val
            
            # Store the forgetting value by cluster
            cluster_logs.setdefault(cid, []).append(forgetting)

        # Update bandit policy with new observations (step 8 in pseudocode)
        if cluster_logs:  # Only update if we have forgetting data
            self.bandit_policy.log_observations({cid: np.array(l) for cid, l in cluster_logs.items()})
        
        return cluster_logs

    def _generate_visualizations(self, strategy):
        """Generate visualizations after each experience"""
        # Cluster visualization
        self.visualize_clusters(
            save_path=os.path.join(self.vis_path, f"experience_{self.current_experience_id}_clusters.png"),
            title=f"Clusters After Experience {self.current_experience_id}"
        )
        
        # Selected samples visualization
        self.visualize_selected_samples(
            save_path=os.path.join(self.vis_path, f"experience_{self.current_experience_id}_selected_samples.png"),
            title=f"Replay Samples After Experience {self.current_experience_id}"
        )
        
        # Enhanced visualizations
        self.viz_tracker.generate_all_plots(
            buffer=self.buffer,
            base_filename=os.path.join(self.vis_path, f"experience_{self.current_experience_id}")
        )
        
        # Final visualizations on last experience
        total_experiences = len(strategy.experience.origin_stream)
        if self.current_experience_id + 1 == total_experiences:
            save_path = os.path.join(self.vis_path, "final_forgetting_dynamics.png")
            self.plot_forgetting_dynamics(save_path=save_path)

            csv_path = os.path.join(self.vis_path, "final_forgetting_log.csv")
            self.save_forgetting_log_csv(csv_path)

    def compute_embeddings(self, X, y):
        """Compute embeddings for input samples (override in subclasses)"""
        # Default: no embeddings, just return inputs
        return X

    def assign_clusters(self, embeddings):
        """Assign clusters to embeddings (must be implemented in subclasses)"""
        raise NotImplementedError("assign_clusters must be implemented in subclasses.")

    def _extract_data_from_dataset(self, dataset):
        """Extract X, y pairs from a dataset"""
        X, y = [], []
        for i in range(len(dataset)):
            sample = dataset[i]
            if isinstance(sample, tuple) and len(sample) >= 2:
                X.append(sample[0].cpu().numpy() if isinstance(sample[0], torch.Tensor) else sample[0])
                y.append(sample[1])
        return np.array(X), np.array(y)
                
    def _track_iteration_forgetting(self, cluster_logs):
        """
        Track forgetting at the iteration level.
        This provides multiple data points per experience for better visualization.
        """
        for cluster_id, forgetting_values in cluster_logs.items():
            if not forgetting_values:
                continue
                
            # Calculate statistics
            mean_forgetting = np.mean(forgetting_values)
            std_forgetting = np.std(forgetting_values)
            
            # Record iteration-level forgetting
            self.iteration_forgetting_log.append({
                'iteration': self.training_iteration,
                'experience_id': self.current_experience_id,
                'exp_iteration': self.current_exp_iterations,
                'cluster_id': cluster_id,
                'mean_forgetting': mean_forgetting,
                'std_forgetting': std_forgetting,
                'min_forgetting': np.min(forgetting_values),
                'max_forgetting': np.max(forgetting_values),
                'num_samples': len(forgetting_values)
            })

    def _save_iteration_forgetting_log(self):
        """Save the iteration-level forgetting log to CSV"""
        if not self.iteration_forgetting_log:
            return
            
        csv_path = os.path.join(self.vis_path, f"iteration_forgetting_log_exp_{self.current_experience_id}.csv")
        
        # Create detailed dataframe
        df = pd.DataFrame(self.iteration_forgetting_log)
        
        # Create a more plot-friendly version
        plot_rows = []
        for row in self.iteration_forgetting_log:
            # Find original experience IDs for this cluster
            orig_exp_ids = set()
            for _, _, _, cid, eid in self.buffer:
                if cid == row['cluster_id']:
                    orig_exp_ids.add(eid)
            
            # For ExperienceAdaptiveReplayPlugin, cluster_id == original_exp_id
            if isinstance(self, ExperienceAdaptiveReplayPlugin):
                orig_exp_ids = {row['cluster_id']}
                
            for orig_exp_id in orig_exp_ids:
                # Skip future experiences
                if orig_exp_id > row['experience_id']:
                    continue
                    
                plot_rows.append({
                    'iteration': row['iteration'],
                    'experience_id': row['experience_id'],
                    'exp_iteration': row['exp_iteration'],
                    'cluster_source': f"Cluster {row['cluster_id']} (Exp {orig_exp_id})",
                    'mean_forgetting': row['mean_forgetting'],
                    'std_forgetting': row['std_forgetting']
                })
                
        plot_df = pd.DataFrame(plot_rows)
        
        # Save both dataframes
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        
        plot_csv_path = csv_path.replace('.csv', '_plot_ready.csv')
        plot_df.to_csv(plot_csv_path, index=False)
        
        print(f"[DEBUG] Iteration-level forgetting logs saved to: {csv_path} and {plot_csv_path}")
    
    def visualize_clusters(self, save_path=None, title="Buffer Clusters Visualization"):
        """Visualizes the buffer clustering using PCA projection."""
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
        """Visualize which samples were selected for replay"""
        if not self.buffer or not hasattr(self, 'selected_replay_embeddings') or len(self.selected_replay_embeddings) == 0:
            print("[DEBUG] No selected replay samples to visualize.")
            return

        # Get embeddings
        all_embeddings = np.vstack([entry[2] for entry in self.buffer])
        selected_embeddings = np.vstack(self.selected_replay_embeddings)
        
        # Ensure both are 2D
        if len(all_embeddings.shape) == 1:
            all_embeddings = all_embeddings.reshape(1, -1)
        if len(selected_embeddings.shape) == 1:
            selected_embeddings = selected_embeddings.reshape(1, -1)

        # Project to 2D using PCA
        pca = PCA(n_components=2)
        all_2d = pca.fit_transform(all_embeddings)
        selected_2d = pca.transform(selected_embeddings)

        # Plot
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
        """Plot forgetting dynamics across experiences"""
        self.viz_tracker.plot_forgetting_dynamics_enhanced(save_path)
        
    def save_forgetting_log_csv(self, csv_path):
        """
        Save the forgetting log as a CSV with properly organized structure for plotting.
        Creates two CSV files:
        1. A detailed CSV with all metrics
        2. A plot-ready simplified CSV for easy visualization
        """
        # Check if we have iteration-level data (from the new tracking)
        if hasattr(self, 'iteration_forgetting_log') and self.iteration_forgetting_log:
            # Save iteration-level data
            iter_csv_path = csv_path.replace('.csv', '_iterations.csv')
            df_iter = pd.DataFrame(self.iteration_forgetting_log)
            
            # Create plot-ready version
            plot_rows = []
            for row in self.iteration_forgetting_log:
                plot_rows.append({
                    "iteration": row['iteration'],
                    "experience": row['experience_id'],
                    "cluster_source": f"Cluster {row['cluster_id']}",
                    "mean_forgetting": row['mean_forgetting'],
                    "std_forgetting": row['std_forgetting']
                })
            
            df_iter_plot = pd.DataFrame(plot_rows)
            
            # Save both CSVs
            iter_plot_csv_path = csv_path.replace('.csv', '_iterations_plot_ready.csv')
            os.makedirs(os.path.dirname(iter_csv_path), exist_ok=True)
            df_iter.to_csv(iter_csv_path, index=False)
            df_iter_plot.to_csv(iter_plot_csv_path, index=False)
            print(f"[DEBUG] Iteration-level forgetting logs saved to: {iter_csv_path} and {iter_plot_csv_path}")
        
        # Also save experience-level data (for backward compatibility)
        if not self.cluster_forgetting_log:
            print("[DEBUG] No experience-level forgetting data to save.")
            return

        # Create rows lists
        detailed_rows = []
        plot_rows = []
        
        # Process each cluster's forgetting data
        for cluster_id, exp_forgetting_data in self.cluster_forgetting_log.items():
            for exp_id, forgetting_values in enumerate(exp_forgetting_data):
                if not forgetting_values:  # Skip empty data
                    continue
                    
                # Calculate statistics
                mean_forgetting = np.mean(forgetting_values)
                std_forgetting = np.std(forgetting_values)
                min_forgetting = np.min(forgetting_values)
                max_forgetting = np.max(forgetting_values)
                num_samples = len(forgetting_values)
                
                # Add to detailed rows
                detailed_rows.append({
                    "experience_id": exp_id,
                    "cluster_id": cluster_id,
                    "mean_forgetting": mean_forgetting,
                    "std_forgetting": std_forgetting,
                    "min_forgetting": min_forgetting,
                    "max_forgetting": max_forgetting,
                    "num_samples": num_samples
                })
                
                # Add to plot rows
                plot_rows.append({
                    "experience": exp_id,
                    "cluster_source": f"Cluster {cluster_id}",
                    "mean_forgetting": mean_forgetting,
                    "std_forgetting": std_forgetting
                })
        
        # Create and save dataframes
        df_detailed = pd.DataFrame(detailed_rows)
        if not df_detailed.empty:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df_detailed.to_csv(csv_path, index=False)
            
            df_plot = pd.DataFrame(plot_rows)
            plot_csv_path = csv_path.replace('.csv', '_plot_ready.csv')
            df_plot.to_csv(plot_csv_path, index=False)
            
            print(f"[DEBUG] Experience-level forgetting logs saved to: {csv_path} and {plot_csv_path}")
        else:
            print("[DEBUG] No valid experience-level forgetting data to save to CSV.")

class ExperienceAdaptiveReplayPlugin(BaseAdaptiveReplayPlugin):
    """Plugin that clusters samples by experience ID"""
    
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

    def assign_clusters(self, embeddings):
        """Assign clusters based on experience ID"""
        return np.array([eid for (_, _, _, _, eid) in self.buffer])
        
    def after_training_exp(self, strategy, **kwargs):
        """Override to update number of clusters based on experience"""
        # Update number of clusters if needed
        current_exp_id = strategy.experience.current_experience
        if current_exp_id >= self.n_clusters:
            # Update the number of clusters
            old_n_clusters = self.n_clusters
            self.n_clusters += 1
            print(f"[DEBUG] Updating clusters from {old_n_clusters} to {self.n_clusters} for experience {current_exp_id}")
        
        # Continue with standard processing
        super().after_training_exp(strategy, **kwargs)


class KMeansInputAdaptiveReplayPlugin(BaseAdaptiveReplayPlugin):
    """Plugin that clusters samples by K-Means on inputs"""
    
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

    def assign_clusters(self, embeddings):
        """Assign clusters using K-Means"""
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        return kmeans.fit_predict(embeddings)
        
    def after_training_exp(self, strategy, **kwargs):
        """Override to update number of clusters based on experience"""
        # Update number of clusters to match the number of experiences
        old_n_clusters = self.n_clusters
        self.n_clusters = strategy.experience.current_experience + 1
        print(f"[DEBUG] Updated to {self.n_clusters} clusters for experience {strategy.experience.current_experience}")
        
        # Don't need to call _init_bandit_policy() here - it's handled in the parent class
        # Continue with standard processing
        super().after_training_exp(strategy, **kwargs)


class TabPFNAdaptiveReplayPlugin(BaseAdaptiveReplayPlugin):
    """
    Adaptive replay that
      • embeds each sample with TabPFNClassifier’s train-encoder,
      • clusters those embeddings,
      • uses a bandit to pick clusters for replay.
    """

    def __init__(self, mem_size=500, batch_size=32,
                 batch_size_mem=5, temperature=1.0, vis_path=None,
                 device="cpu"):
        if TabPFNClassifier is None:
            raise ImportError("Install tabpfn to use this plugin.")

        super().__init__(
            mem_size=mem_size,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            temperature=temperature,
            n_clusters=1,
            vis_path=vis_path,
        )

        self.device = device
        self.tabpfn_model = TabPFNClassifier(device=device)

    # -----------------------------------------------------------------
    #            E M B E D D I N G S
    # -----------------------------------------------------------------
    @torch.no_grad()
    def compute_embeddings(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit TabPFNClassifier on the current mini-batch and return the
        hidden-state embeddings (one per sample). Falls back to X if
        TabPFN fails for some reason.
        """
        try:
            self.tabpfn_model.fit(X, y)
            # TabPFN exposes embeddings via helper
            embeddings = self.tabpfn_model.get_embeddings(X, data_source="train")
            return embeddings
        except Exception as e:                 # graceful degradation
            print(f"[TabPFN-emb] fallback to raw inputs: {e}")
            return X.astype(np.float32)

    # -----------------------------------------------------------------
    #            C L U S T E R I N G
    # -----------------------------------------------------------------
    def assign_clusters(self, embeddings: np.ndarray) -> np.ndarray:
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        return kmeans.fit_predict(embeddings)

    # -----------------------------------------------------------------
    #   B A N D I T   R E S E T   A F T E R   E A C H   E X P
    # -----------------------------------------------------------------
    def after_training_exp(self, strategy, **kwargs):
        self.n_clusters = strategy.experience.current_experience + 1
        self._init_bandit_policy()
        super().after_training_exp(strategy, **kwargs)

    # -----------------------------------------------------------------
    #     (optional) override loss per cluster to CrossEntropy
    # -----------------------------------------------------------------
    def _compute_losses_per_cluster(self, strategy):
        """
        Compute mean cross-entropy loss for each cluster in the current minibatch.
        This keeps the exact signature the base class expects.
        """
        # 1. Grab the network’s logits / targets from the strategy
        logits   = strategy.mb_output          # shape [B, C]
        targets  = strategy.mb_y.long()        # shape [B]

        # 2. Determine which cluster each sample belongs to
        #    (you already stored this mapping in self.current_batch_clusters
        #     right after `assign_clusters(...)` in before_training_iteration)
        clusters = self.current_batch_clusters  # shape [B]

        ce = torch.nn.functional.cross_entropy
        losses = {}
        for cid in range(self.n_clusters):
            mask = clusters == cid
            if mask.any():
                losses[cid] = [ce(logits[mask], targets[mask]).item()]
            else:
                losses[cid] = [0.0]

        return losses        # dict {cid: [loss_value]}


__all__ = [
    "BaseAdaptiveReplayPlugin",
    "ExperienceAdaptiveReplayPlugin",
    "KMeansInputAdaptiveReplayPlugin",
    "TabPFNAdaptiveReplayPlugin"
]

if __name__ == "__main__":
    print("Adaptive Replay Plugins Loaded.")