import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from avalanche.training.plugins import SupervisedPlugin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os, sys, random
from collections import Counter

# Bandit imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "BanditCL"))
from NSMAB.Models.MovingAverage import MovingAverage
from NSMAB.Policies.Policies import NSMAB_Policy_wesu
from NSMAB.Samplers import BoltzmannSampling

class ParametricAdaptiveReplayExpClustersPlugin(SupervisedPlugin):
    def __init__(self, mem_size=500, batch_size=32, temperature=1.0, vis_path=None):
        super().__init__()
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.temperature = temperature
        self.vis_path = vis_path

        self.buffer = []  # Stores (x, y, cluster_id, experience_id)
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.scaler = StandardScaler()

        modeller_ctor = lambda _: MovingAverage(alpha=0.7, assumed_increase_in_mean=10.0)
        sampling_strategy = BoltzmannSampling(softmax_temperature=self.temperature)
        self.bandit_policy = NSMAB_Policy_wesu(modeller_ctor, sampling_strategy)

    def visualize_embeddings(self, X, labels, title="Replay Buffer Clusters"):
        if self.vis_path is None:
            return
        os.makedirs(self.vis_path, exist_ok=True)
        filepath = os.path.join(self.vis_path, f"{title.replace(' ', '_')}.png")
        X_scaled = self.scaler.fit_transform(X)
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(X_scaled)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Experience/Cluster ID')
        plt.title(title)
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.savefig(filepath)
        plt.close()
        print(f"[ReplayPlugin] Embedding visualization saved to: {filepath}")

    def visualize_clusters(self, X, cluster_ids, selected_sample_indices, title='Selected Samples'):
        if self.vis_path is None:
            return
        os.makedirs(self.vis_path, exist_ok=True)
        filepath = os.path.join(self.vis_path, f"{title.replace(' ', '_')}.png")
        X_scaled = self.scaler.fit_transform(X)
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(X_scaled)
        colors = np.array(["#cccccc"] * len(X))
        for i in selected_sample_indices:
            if i < len(colors):
                colors[i] = "red"
        plt.figure(figsize=(8, 8))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.7)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Replayed Samples'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#cccccc', markersize=10, label='Non-replayed')
        ]
        plt.legend(handles=legend_elements)
        plt.title(title)
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.savefig(filepath)
        plt.close()
        print(f"[ReplayPlugin] Replay visualization saved to: {filepath}")

    def before_training_exp(self, strategy, **kwargs):
        self.current_experience_id = strategy.experience.current_experience
        print(f"[ReplayPlugin] Processing experience {self.current_experience_id}")
        if self.current_experience_id == 0 or not self.buffer:
            print("[ReplayPlugin] Skipping replay.")
            return

        past_buffer = [(x, y, cid, eid) for (x, y, cid, eid) in self.buffer if eid < self.current_experience_id]
        print(f"[ReplayPlugin] Past buffer size: {len(past_buffer)}")

        past_x = np.stack([x for (x, _, _, _) in past_buffer])
        past_cids = [cid for (_, _, cid, _) in past_buffer]
        if self.vis_path:
            self.visualize_embeddings(past_x, past_cids, title=f"Experience_{self.current_experience_id}_Past_Buffer")

        cluster_to_samples = {}
        for x, y, cid, _ in past_buffer:
            cluster_to_samples.setdefault(cid, []).append((x, y))

        self.bandit_policy.add_clusters(1)  # Add a new cluster for current experience
        k = min(self.batch_size, len(cluster_to_samples))
        bandit_selected = self.bandit_policy.suggest_clusters_to_sample_from(k)

        cluster_counts = Counter(bandit_selected)
        for cid in cluster_to_samples:
            if cid not in cluster_counts:
                cluster_counts[cid] = 1
        total_weight = sum(cluster_counts.values())
        samples_per_cluster = {
            cid: max(1, int((cluster_counts[cid] / total_weight) * self.mem_size))
            for cid in cluster_counts
        }

        selected_x, selected_y, selected_ids = [], [], []
        selected_sample_indices = []
        past_index_map = {id(x): i for i, (x, _, _, _) in enumerate(past_buffer)}
        for cid, samples in cluster_to_samples.items():
            n = min(len(samples), samples_per_cluster.get(cid, 0))
            chosen = random.sample(samples, n)
            for x, y in chosen:
                selected_x.append(x)
                selected_y.append(y)
                selected_ids.append(cid)
                idx = past_index_map.get(id(x), None)
                if idx is not None:
                    selected_sample_indices.append(idx)

        if self.vis_path:
            self.visualize_clusters(past_x, past_cids, selected_sample_indices, title=f"Experience_{self.current_experience_id}_Selected_Replay")

        current_x, current_y = self._extract_data_from_dataset(strategy.experience.dataset)
        combined_x = np.concatenate([current_x, np.stack(selected_x)])
        combined_y = np.concatenate([current_y, np.stack(selected_y)])
        combined_ids = np.concatenate([np.full(len(current_x), self.current_experience_id), np.array(selected_ids)])

        full_dataset = TensorDataset(torch.tensor(combined_x), torch.tensor(combined_y), torch.tensor(combined_ids))
        strategy.adapted_dataset = full_dataset
        strategy.dataloader = DataLoader(full_dataset, batch_size=32, shuffle=True, drop_last=True)
        print(f"[ReplayPlugin] Injected {len(selected_x)} replay samples into dataloader.")

    def before_training_iteration(self, strategy, **kwargs):
        if self.current_experience_id == 0: return
        with torch.no_grad():
            outputs = strategy.model(strategy.mb_x)
            losses = self.criterion(outputs.squeeze(), strategy.mb_y.squeeze())
            strategy.mb_loss = losses.detach().cpu().numpy()

    def after_training_exp(self, strategy, **kwargs):
        X, y = self._extract_data_from_dataset(strategy.experience.dataset)
        current_exp_id = strategy.experience.current_experience

        # Assign experience as cluster
        new_entries = [(x, y_val, current_exp_id, current_exp_id) for x, y_val in zip(X, y)]
        self.buffer.extend(new_entries)

        # Rebalance by experience
        all_exp_ids = sorted(set(eid for (_, _, _, eid) in self.buffer))
        samples_per_exp = self.mem_size // len(all_exp_ids)
        rebalanced = []
        for eid in all_exp_ids:
            exp_samples = [entry for entry in self.buffer if entry[3] == eid]
            if len(exp_samples) > samples_per_exp:
                exp_samples = random.sample(exp_samples, samples_per_exp)
            rebalanced.extend(exp_samples)
        self.buffer = rebalanced

        # Log updated losses
        X_buf = np.stack([x for (x, _, _, _) in self.buffer])
        y_buf = np.stack([y for (_, y, _, _) in self.buffer])
        cid_buf = [cid for (_, _, cid, _) in self.buffer]
        with torch.no_grad():
            preds = strategy.model(torch.tensor(X_buf).to(strategy.device))
            losses = self.criterion(preds.squeeze(), torch.tensor(y_buf.squeeze()).to(strategy.device)).cpu().numpy()

        cluster_logs = {}
        for cid, loss in zip(cid_buf, losses):
            cluster_logs.setdefault(cid, []).append(loss)
            
        # Get max cluster ID (i.e., highest experience ID in buffer)
        max_cid = max(entry[2] for entry in self.buffer)
        if max_cid >= self.bandit_policy.num_clusters:
            num_new = max_cid - self.bandit_policy.num_clusters + 1
            self.bandit_policy.add_clusters(num_new)
    
        print(f"[ReplayPlugin] Bandit now tracking {self.bandit_policy.num_clusters} clusters.")
   
        self.bandit_policy.log_observations({cid: np.array(v) for cid, v in cluster_logs.items()})
        self.bandit_policy.next_timestep()
        print(f"[ReplayPlugin] Buffer updated. Experiences now act as clusters. Total: {len(self.buffer)} samples.")

    def _extract_data_from_dataset(self, dataset):
        X, y = [], []
        for i in range(len(dataset)):
            sample = dataset[i]
            if isinstance(sample, tuple) and len(sample) >= 2:
                x_item = sample[0].cpu().numpy() if isinstance(sample[0], torch.Tensor) else sample[0]
                X.append(x_item)
                y.append(sample[1])
        return np.array(X), np.array(y)

__all__ = ["ParametricAdaptiveReplayExpClustersPlugin"]

if __name__ == "__main__":
    print("Parametric Adaptive Replay Plugin for Avalanche")
    print("------------------------------------------")
    print("Import this module to use the ParametricAdaptiveReplayPlugin in your project.")
    print("Example usage: from parametric_replay import parametric_adaptive_replay")

