# """
# Parametric Adaptive Replay Strategy for Avalanche Continual Learning Framework

# This module implements a continual learning replay strategy based on direct clustering 
# of input data and bandit-based selection, integrated with Avalanche's framework.
# """

# import numpy as np
# import torch
# from sklearn.cluster import KMeans
# from avalanche.training.plugins import SupervisedPlugin
# from torch.utils.data import DataLoader, TensorDataset

# # Add BanditCL to sys.path
# import sys
# import os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# banditcl_path = os.path.join(current_dir, "BanditCL")
# sys.path.append(banditcl_path)

# from NSMAB.Models.MovingAverage import MovingAverage
# from NSMAB.Policies.Policies import NSMAB_Policy_Compositional
# from NSMAB.Samplers import BoltzmannSampling

# import torch.nn.functional as F
# import random
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# class ParametricAdaptiveReplayPlugin(SupervisedPlugin):
#     """
#     Avalanche plugin that implements a replay strategy based on direct clustering
#     of input data features and bandit-based selection of replay samples.
#     """
    
#     def __init__(self, mem_size=500, batch_size=32, temperature=1.0, n_clusters=4, vis_path=None):
#         """
#         Initialize the parametric adaptive replay plugin.
        
#         Args:
#             mem_size: Maximum number of samples to keep in memory
#             batch_size: Size of the replay batch
#             temperature: Temperature parameter for Boltzmann sampling
#             n_clusters: Number of clusters to create
#             vis_path: Path to save visualizations (None to disable)
#         """
#         super().__init__()
#         self.mem_size = mem_size
#         self.batch_size = batch_size
#         self.temperature = temperature
#         self.n_clusters = n_clusters
#         self.vis_path = vis_path

#         # Buffer stores (x, y, cluster_id, experience_id)
#         self.buffer = []
#         self.kmeans = None
#         self.scaler = StandardScaler()
#         self.criterion = torch.nn.MSELoss(reduction='none')

#         # Initialize bandit policy
#         modeller_ctor = lambda _: MovingAverage(alpha=0.7, assumed_increase_in_mean=10.0)
#         sampling_strategy = BoltzmannSampling(softmax_temperature=self.temperature)
#         self.bandit_policy = NSMAB_Policy_Compositional(modeller_ctor, sampling_strategy)
#         self.bandit_policy.add_clusters(self.n_clusters)
        
#     def cluster_data(self, X):
#         """
#         Cluster the input data directly.
        
#         Args:
#             X: Input features (numpy array or torch tensor)
            
#         Returns:
#             Cluster assignments for each sample
#         """
#         if isinstance(X, torch.Tensor):
#             X_np = X.cpu().numpy()
#         else:
#             X_np = X
            
#         # Apply scaling to normalize features
#         X_scaled = self.scaler.fit_transform(X_np)
        
#         # Ensure we don't try to create more clusters than samples
#         n_clusters = min(self.n_clusters, len(X_scaled))
        
#         self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#         return self.kmeans.fit_predict(X_scaled)
    
#     def visualize_data(self, X, labels, title="Data Clusters"):
#         """
#         Visualize clustered data using PCA.
        
#         Args:
#             X: Input features 
#             labels: Corresponding cluster assignments
#             title: Plot title
            
#         Returns:
#             filepath: Path to saved visualization file
#         """
#         # Skip visualization if path not provided
#         if self.vis_path is None:
#             return None
        
#         # Create directory if needed
#         os.makedirs(self.vis_path, exist_ok=True)
#         filepath = os.path.join(self.vis_path, f"{title.replace(' ', '_')}.png")
        
#         # Scale data and reduce to 2D for visualization
#         X_scaled = self.scaler.transform(X)
#         pca = PCA(n_components=2)
#         data_2d = pca.fit_transform(X_scaled)
        
#         plt.figure(figsize=(10, 8))
#         scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
#         plt.colorbar(scatter, label='Cluster')
#         plt.title(title)
#         plt.xlabel('PCA Component 1')
#         plt.ylabel('PCA Component 2')
        
#         # Save and close
#         plt.savefig(filepath)
#         plt.close()
        
#         print(f"Visualization saved to: {filepath}")
#         return filepath
    
#     # def visualize_clusters(self, X, cluster_ids, selected_clusters, title='Selected Clusters'):
#     #     """
#     #     Visualize selected clusters.
        
#     #     Args:
#     #         X: Input features
#     #         cluster_ids: Cluster assignments
#     #         selected_clusters: IDs of selected clusters
#     #         title: Plot title
#     #     """
#     #     if self.vis_path is None:
#     #         return
            
#     #     X_scaled = self.scaler.transform(X)
#     #     pca = PCA(n_components=2)
#     #     reduced = pca.fit_transform(X_scaled)
        
#     #     plt.figure(figsize=(10, 8))
#     #     colors = ["#cccccc"] * len(X)
#     #     for i, cid in enumerate(cluster_ids):
#     #         if cid in selected_clusters:
#     #             colors[i] = "red"
#     #     plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, s=15, alpha=0.8)
#     #     plt.title(title)
                
#     #     os.makedirs(self.vis_path, exist_ok=True)
#     #     filepath = os.path.join(self.vis_path, f"{title.replace(' ', '_')}.png")
        
#     #     plt.savefig(filepath)
#     #     plt.close()
#     #     print(f"[ParametricReplayPlugin] Saved cluster visualization to {filepath}")
    
#     def visualize_clusters(self, X, cluster_ids, selected_clusters, title='Selected Clusters'):
#         """
#         Visualize selected clusters with consistent style matching the embeddings visualization.
        
#         Args:
#             X: Embedding vectors
#             cluster_ids: Cluster assignments for each sample
#             selected_clusters: IDs of clusters selected by the bandit
#             title: Plot title
#         """
#         # Skip visualization if path not provided
#         if self.vis_path is None:
#             return None
        
#         # Create directory if needed
#         os.makedirs(self.vis_path, exist_ok=True)
#         filepath = os.path.join(self.vis_path, f"{title.replace(' ', '_')}.png")
        
#         # Use the same PCA reduction as in visualize_embeddings
#         X_scaled = self.scaler.transform(X)
#         pca = PCA(n_components=2)
#         reduced = pca.fit_transform(X_scaled)
        
#         # Create a figure with the same size and style as the embeddings visualization
#         plt.figure(figsize=(8, 8))
        
#         # Color points based on selection (red for selected, gray for non-selected)
#         colors = np.array(["#cccccc"] * len(X_scaled))
#         for i, cid in enumerate(cluster_ids):
#             if cid in selected_clusters:
#                 colors[i] = "red"
        
#         # Use the exact same scatter plot parameters for consistency
#         plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.7)
        
#         # Add title and labels with the same styling
#         plt.title(title)
#         plt.xlabel('PCA Component 1')
#         plt.ylabel('PCA Component 2')
        
#         # Add legend to explain colors
#         from matplotlib.lines import Line2D
#         legend_elements = [
#             Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Selected Clusters'),
#             Line2D([0], [0], marker='o', color='w', markerfacecolor='#cccccc', markersize=10, label='Non-selected Clusters')
#         ]
#         plt.legend(handles=legend_elements)
        
#         # Add information about selection
#         selected_count = np.sum(np.isin(cluster_ids, selected_clusters))
#         total_count = len(cluster_ids)
#         percentage = 100 * selected_count / total_count if total_count > 0 else 0
#         selected_clusters_str = ', '.join(map(str, selected_clusters))
#         plt.figtext(0.5, 0.01, 
#                     f"Selected clusters: {selected_clusters_str}\n"
#                     f"Selected points: {selected_count}/{total_count} ({percentage:.1f}%)",
#                     ha='center', fontsize=10)
        
#         # Save and close
#         plt.savefig(filepath)
#         plt.close()
        
#         print(f"[BanditReplayPlugin] Selected clusters visualization saved to: {filepath}")
#         return filepath

#     def before_training_exp(self, strategy, **kwargs):
#         """
#         Called before training on an experience.
        
#         Implements replay by selecting samples from memory using bandit policy
#         and combining them with current experience data.
#         """
#         self.current_experience_id = strategy.experience.current_experience
#         print(f"[ParametricReplayPlugin] Processing experience {self.current_experience_id}")

#         # First experience - no replay needed
#         if self.current_experience_id == 0:
#             print("[ParametricReplayPlugin] Experience 0 — using default dataloader, no replay.")
#             return
        
#         # Reset bandit policy to prevent arm accumulation
#         modeller_ctor = lambda _: MovingAverage(alpha=0.7, assumed_increase_in_mean=10.0)
#         sampling_strategy = BoltzmannSampling(softmax_temperature=self.temperature)
#         self.bandit_policy = NSMAB_Policy_Compositional(modeller_ctor, sampling_strategy)
#         self.bandit_policy.add_clusters(self.n_clusters)
            
#         if not self.buffer:
#             print("[ParametricReplayPlugin] Warning: Memory buffer is empty.")
#             return

#         # Filter buffer to previous experiences only
#         past_buffer = [(x, y, cid, eid) for (x, y, cid, eid) in self.buffer 
#                        if eid < self.current_experience_id]
#         print(f"[ParametricReplayPlugin] Buffer size: {len(self.buffer)}")
#         print(f"[ParametricReplayPlugin] Past buffer size: {len(past_buffer)}")
        
#         if not past_buffer:
#             print("[ParametricReplayPlugin] Warning: No valid memory samples for replay.")
#             return

#         # Group samples by cluster ID
#         cluster_to_samples = {i: [] for i in range(self.n_clusters)}
#         for x, y, cid, eid in past_buffer:
#             cluster_to_samples.setdefault(cid, []).append((x, y))

#         available_cluster_ids = list(range(self.n_clusters))
#         print(f"[ParametricReplayPlugin] Available cluster IDs: {available_cluster_ids}")
        
#         k = min(self.batch_size, len(available_cluster_ids))
#         selected_cluster_idxs = self.bandit_policy.suggest_clusters_to_sample_from(k)
#         print(f"[ParametricReplayPlugin] Selected cluster indices from bandit: {selected_cluster_idxs}")

#         # Map cluster index (from policy) back to actual cluster ID
#         if max(selected_cluster_idxs, default=-1) >= len(available_cluster_ids):
#             print(f"[ParametricReplayPlugin] Warning: Bandit returned invalid cluster index. "
#                   f"Available: {len(available_cluster_ids)}, Got: {selected_cluster_idxs}")
#             return
            
#         selected_clusters = [available_cluster_ids[i] for i in selected_cluster_idxs]
        
#         # Visualize selected clusters if enabled
#         if self.vis_path is not None:
#             replay_x = np.stack([x for (x, _, _, eid) in self.buffer if eid != self.current_experience_id])
#             replay_cids = [cid for (_, _, cid, eid) in self.buffer if eid != self.current_experience_id]
            
#             if len(replay_x) > 0 and len(replay_cids) > 0:
#                 self.visualize_clusters(
#                     X=replay_x,
#                     cluster_ids=replay_cids,
#                     selected_clusters=selected_clusters,
#                     title=f"Experience_{self.current_experience_id}_Selected_Clusters"
#                 )
            
#         # Select samples from each chosen cluster
#         selected_x, selected_y, selected_ids = [], [], []
#         for cid in selected_clusters:
#             samples = cluster_to_samples.get(cid, [])
#             if not samples:
#                 print(f"[ParametricReplayPlugin] Skipping cluster {cid} — no samples available.")
#                 continue
#             num_samples = min(len(samples), max(1, 550 // k))
#             chosen = random.sample(samples, num_samples)
#             for x, y in chosen:
#                 selected_x.append(x)
#                 selected_y.append(y)
#                 selected_ids.append(cid)

#         selected_x = np.stack(selected_x)
#         selected_y = np.stack(selected_y)

#         # Create combined dataset with current data and replay samples
#         current_dataset = strategy.experience.dataset
#         current_x, current_y = self._extract_data_from_dataset(current_dataset)

#         combined_x = np.concatenate([current_x, selected_x])
#         combined_y = np.concatenate([current_y, selected_y])
#         combined_ids = np.concatenate([np.full(len(current_x), self.current_experience_id), np.array(selected_ids)])

#         tensor_x = torch.tensor(combined_x)
#         tensor_y = torch.tensor(combined_y)
#         task_ids = torch.tensor(combined_ids)

#         full_dataset = TensorDataset(tensor_x, tensor_y, task_ids)
#         strategy.adapted_dataset = full_dataset
#         strategy.dataloader = DataLoader(
#                 full_dataset,
#                 batch_size=64,
#                 shuffle=True,
#                 drop_last=True
#             )

#         print(f"[ParametricReplayPlugin] Injected {len(selected_x)} replay samples "
#               f"from {len(selected_clusters)} selected clusters into strategy.dataloader")

#     def before_training_iteration(self, strategy, **kwargs):
#         """
#         Called before each training iteration.
        
#         Calculates per-sample losses for bandit feedback.
#         """
#         self.current_experience_id = strategy.experience.current_experience
#         if self.current_experience_id == 0:
#             return
        
#         with torch.no_grad():
#             outputs = strategy.model(strategy.mb_x)
#             losses = self.criterion(outputs.squeeze(), strategy.mb_y.squeeze())
#             strategy.mb_loss = losses.detach().cpu().numpy()

#     def after_training_exp(self, strategy, **kwargs):
#         """
#         Called after training on an experience.
        
#         Updates memory buffer with samples from current experience
#         and provides feedback to bandit policy.
#         """
#         # Ensure bandit policy has the right number of arms
#         self.bandit_policy.add_clusters(self.n_clusters)
        
#         # Get current experience data
#         dataset = strategy.experience.dataset
#         X, y = self._extract_data_from_dataset(dataset)
        
#         # Directly cluster the input data (no TabPFN embedding)
#         cluster_ids = self.cluster_data(X)
        
#         # Visualize the clustered data if visualization is enabled
#         if self.vis_path is not None:
#             self.visualize_data(
#                 X,
#                 cluster_ids,
#                 title=f"Experience_{self.current_experience_id}_Clusters"
#             )

#         # Sample a subset for loss estimation (to save computation)
#         sample_size = max(1, int(0.2 * len(X)))
#         idxs = np.random.choice(len(X), sample_size, replace=False)
#         X_sample = X[idxs]
#         y_sample = y[idxs]
#         cid_sample = cluster_ids[idxs]

#         print(f"[ParametricReplayPlugin] Using {sample_size} samples for loss estimation.")

#         # Calculate loss for each sample in the selected subset
#         with torch.no_grad():
#             model = strategy.model.to(strategy.device)
#             x_tensor = torch.tensor(X_sample).to(strategy.device)
#             y_tensor = torch.tensor(y_sample).to(strategy.device)
#             preds = model(x_tensor)
#             losses = self.criterion(preds.squeeze(), y_tensor.squeeze()).cpu().numpy()

#         # Store samples in buffer and provide feedback to bandit
#         current_exp_id = strategy.experience.current_experience
#         for x, y_val, cid, loss in zip(X_sample, y_sample, cid_sample, losses):
#             self.buffer.append((x, y_val, cid, current_exp_id))
#             self.bandit_policy.log_observations({cid: np.array([loss])})

#         # # Limit buffer size if needed
#         # if len(self.buffer) > self.mem_size:
#         #     self.buffer = self.buffer[-self.mem_size:]
            
#         print(f"[ParametricReplayPlugin] Updated buffer with samples from experience "
#               f"{current_exp_id}. Buffer size: {len(self.buffer)}")

#     def _extract_data_from_dataset(self, dataset):
#         """
#         Extract data from an Avalanche dataset.
        
#         Args:
#             dataset: Avalanche dataset
            
#         Returns:
#             X, y: Extracted features and targets
#         """
#         X, y, task_labels = [], [], []
        
#         # Handle dataset types
#         if hasattr(dataset, "_flat_data_x"):
#             # It's a flat Avalanche dataset with direct access to data
#             X = dataset._flat_data_x
#             y = dataset._flat_data_y
#             if hasattr(dataset, "_flat_task_labels"):
#                 task_labels = dataset._flat_task_labels
#             return X, y
        
#         # Try to extract data iteratively
#         for i in range(len(dataset)):
#             sample = dataset[i]
#             if isinstance(sample, tuple):
#                 if len(sample) >= 2:
#                     x_item = sample[0].cpu().numpy() if isinstance(sample[0], torch.Tensor) else sample[0]
#                     X.append(x_item)
#                     y.append(sample[1])
#                 if len(sample) >= 3:
#                     task_labels.append(sample[2])
        
#         X = np.array(X)
#         y = np.array(y)
#         return X, y

##############################################################################################################################################################################################

import numpy as np
import torch
from sklearn.cluster import KMeans
from avalanche.training.plugins import SupervisedPlugin
from torch.utils.data import DataLoader, TensorDataset
import os, sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random

# Bandit imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "BanditCL"))
from NSMAB.Models.MovingAverage import MovingAverage
from NSMAB.Policies.Policies import NSMAB_Policy_Compositional, NSMAB_Policy_wesu
from NSMAB.Samplers import BoltzmannSampling

class ParametricAdaptiveReplayPlugin(SupervisedPlugin):
    def __init__(self, mem_size=500, batch_size=32, temperature=1.0, n_clusters=4, vis_path=None):
        super().__init__()
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.temperature = temperature
        self.n_clusters = n_clusters
        self.vis_path = vis_path

        self.buffer = []  # Stores (x, y, cluster_id, experience_id)
        self.scaler = StandardScaler()
        self.criterion = torch.nn.MSELoss(reduction='none')

        modeller_ctor = lambda _: MovingAverage(alpha=0.7, assumed_increase_in_mean=10.0)
        sampling_strategy = BoltzmannSampling(softmax_temperature=self.temperature)
        # self.bandit_policy = NSMAB_Policy_Compositional(modeller_ctor, sampling_strategy)
        self.bandit_policy = NSMAB_Policy_wesu(modeller_ctor, sampling_strategy)
        self.bandit_policy.add_clusters(self.n_clusters)

    def cluster_data(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans = KMeans(n_clusters=min(self.n_clusters, len(X)), random_state=42)
        return self.kmeans.fit_predict(X_scaled)

    def visualize_embeddings(self, X, labels, title="Replay Buffer Clusters"):
        """
        Visualize raw input samples using PCA, colored by cluster ID.
        """
        if self.vis_path is None:
            return

        os.makedirs(self.vis_path, exist_ok=True)
        filepath = os.path.join(self.vis_path, f"{title.replace(' ', '_')}.png")

        X_scaled = self.scaler.transform(X)
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(X_scaled)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(title)
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.savefig(filepath)
        plt.close()

        print(f"[ReplayPlugin] Embedding visualization saved to: {filepath}")
        return filepath

    def visualize_clusters(self, X, cluster_ids, selected_sample_indices, title='Selected Samples'):
        """
        Visualize samples in buffer, highlighting selected samples.
        """
        if self.vis_path is None:
            return

        os.makedirs(self.vis_path, exist_ok=True)
        filepath = os.path.join(self.vis_path, f"{title.replace(' ', '_')}.png")

        X_scaled = self.scaler.transform(X)
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
        return filepath

    def before_training_exp(self, strategy, **kwargs):
        self.current_experience_id = strategy.experience.current_experience
        print(f"[ReplayPlugin] Processing experience {self.current_experience_id}")

        if self.current_experience_id == 0:
            print("[ReplayPlugin] Experience 0 — no replay.")
            return

        if not self.buffer:
            print("[ReplayPlugin] Warning: Buffer empty.")
            return

        # === Extract past buffer only ===
        past_buffer = [(x, y, cid, eid) for (x, y, cid, eid) in self.buffer if eid < self.current_experience_id]
        print(f"[ReplayPlugin] Past buffer size: {len(past_buffer)}")

        # === Visualization of buffer cluster structure ===
        if self.vis_path:
            past_x = np.stack([x for (x, _, _, _) in past_buffer])
            past_cids = [cid for (_, _, cid, _) in past_buffer]
            self.visualize_embeddings(past_x, past_cids, title=f"Experience_{self.current_experience_id}_Past_Buffer_Clusters")

        # === Group buffer by cluster ===
        cluster_to_samples = {i: [] for i in range(self.n_clusters)}
        for x, y, cid, eid in past_buffer:
            cluster_to_samples[cid].append((x, y))

        # === Bandit selects clusters ===
        k = min(self.batch_size, self.n_clusters)
        bandit_selected = self.bandit_policy.suggest_clusters_to_sample_from(k)

        # # === Ensure all clusters are covered ===
        selected_clusters = set(bandit_selected)
        # for c in range(self.n_clusters):
        #     selected_clusters.add(c)
        # selected_clusters = list(selected_clusters)

        # === Proportional sample allocation ===
        from collections import Counter
        cluster_counts = Counter(bandit_selected)
        for cid in selected_clusters:
            if cid not in cluster_counts:
                cluster_counts[cid] = 1
        total_weight = sum(cluster_counts.values())
        samples_per_cluster = {
            cid: max(1, int((cluster_counts[cid] / total_weight) * self.mem_size))
            for cid in selected_clusters
        }

        print(f"[ReplayPlugin] Selected clusters: {selected_clusters}")

        selected_x, selected_y, selected_ids = [], [], []
        selected_sample_indices = []
        past_index_map = {id(x): i for i, (x, _, _, _) in enumerate(past_buffer)}

        for cid in selected_clusters:
            samples = cluster_to_samples.get(cid, [])
            if not samples: continue
            num = min(len(samples), samples_per_cluster[cid])
            chosen = random.sample(samples, num)
            for x, y in chosen:
                selected_x.append(x)
                selected_y.append(y)
                selected_ids.append(cid)
                # Track index before copy
                idx = past_index_map.get(id(x), None)
                if idx is not None:
                    selected_sample_indices.append(idx)
                    
        
        if self.vis_path:
            self.visualize_clusters(
                X=past_x,
                cluster_ids=past_cids,
                selected_sample_indices=selected_sample_indices,
                title=f"Experience_{self.current_experience_id}_Selected_Replay"
            )

        # === Combine current + replay for training ===
        current_x, current_y = self._extract_data_from_dataset(strategy.experience.dataset)
        combined_x = np.concatenate([current_x, selected_x])
        combined_y = np.concatenate([current_y, selected_y])
        combined_ids = np.concatenate([np.full(len(current_x), self.current_experience_id), np.array(selected_ids)])

        # === Setup dataset and dataloader ===
        full_dataset = TensorDataset(
            torch.tensor(combined_x),
            torch.tensor(combined_y),
            torch.tensor(combined_ids)
        )

        strategy.adapted_dataset = full_dataset
        strategy.dataloader = DataLoader(full_dataset, batch_size=64, shuffle=True, drop_last=True)

        print(f"[ReplayPlugin] Injected {len(selected_x)} replay samples into dataloader")

    def before_training_iteration(self, strategy, **kwargs):
        if self.current_experience_id == 0: return
        with torch.no_grad():
            outputs = strategy.model(strategy.mb_x)
            losses = self.criterion(outputs.squeeze(), strategy.mb_y.squeeze())
            strategy.mb_loss = losses.detach().cpu().numpy()

    def after_training_exp(self, strategy, **kwargs):
        X, y = self._extract_data_from_dataset(strategy.experience.dataset)
        current_exp_id = strategy.experience.current_experience

        # Add all new samples (cluster ID will be re-assigned)
        new_entries = [(x, y_val, -1, current_exp_id) for x, y_val in zip(X, y)]
        self.buffer.extend(new_entries)

        # Rebalance buffer evenly across experiences
        all_exp_ids = sorted(set(eid for (_, _, _, eid) in self.buffer))
        samples_per_exp = self.mem_size // len(all_exp_ids)
        rebalanced = []
        for eid in all_exp_ids:
            exp_samples = [entry for entry in self.buffer if entry[3] == eid]
            if len(exp_samples) > samples_per_exp:
                exp_samples = random.sample(exp_samples, samples_per_exp)
            rebalanced.extend(exp_samples)
        self.buffer = rebalanced

        # Recluster entire buffer using raw input
        all_x = np.stack([x for (x, _, _, _) in self.buffer])
        cluster_ids = self.cluster_data(all_x)
        self.buffer = [
            (x, y, cid, eid) for (cid, (x, y, _, eid)) in zip(cluster_ids, self.buffer)
        ]

        # Compute updated loss over buffer samples
        y_buf = np.stack([y for (_, y, _, _) in self.buffer])
        cid_buf = [cid for (_, _, cid, _) in self.buffer]
        with torch.no_grad():
            model = strategy.model.to(strategy.device)
            preds = model(torch.tensor(all_x).to(strategy.device))
            losses = self.criterion(preds.squeeze(), torch.tensor(y_buf.squeeze()).to(strategy.device)).cpu().numpy()

        # Log updated losses to bandit
        cluster_logs = {}
        for cid, loss in zip(cid_buf, losses):
            cluster_logs.setdefault(cid, []).append(loss)
        self.bandit_policy.log_observations({cid: np.array(v) for cid, v in cluster_logs.items()})
        self.bandit_policy.next_timestep()

        print(f"[ReplayPlugin] Buffer updated with reclustered data (size = {len(self.buffer)})")

    def _extract_data_from_dataset(self, dataset):
        """
        Extract data from an Avalanche dataset.
        
        Args:
            dataset: Avalanche dataset
            
        Returns:
            X, y: Extracted features and targets
        """
        X, y, task_labels = [], [], []
        
        # Handle dataset types
        if hasattr(dataset, "_flat_data_x"):
            # It's a flat Avalanche dataset with direct access to data
            X = dataset._flat_data_x
            y = dataset._flat_data_y
            if hasattr(dataset, "_flat_task_labels"):
                task_labels = dataset._flat_task_labels
            return X, y
        
        # Try to extract data iteratively
        for i in range(len(dataset)):
            sample = dataset[i]
            if isinstance(sample, tuple):
                if len(sample) >= 2:
                    x_item = sample[0].cpu().numpy() if isinstance(sample[0], torch.Tensor) else sample[0]
                    X.append(x_item)
                    y.append(sample[1])
                if len(sample) >= 3:
                    task_labels.append(sample[2])
        
        X = np.array(X)
        y = np.array(y)
        return X, y


def parametric_adaptive_replay(mem_size=500, batch_size=32, temperature=1.0, n_clusters=4, vis_path=None):
    """
    Factory function to create a ParametricAdaptiveReplayPlugin instance.
    
    Args:
        mem_size: Maximum number of samples to keep in memory
        batch_size: Size of the replay batch
        temperature: Temperature parameter for Boltzmann sampling
        n_clusters: Number of clusters to create
        vis_path: Path to save visualizations
        
    Returns:
        ParametricAdaptiveReplayPlugin instance
    """
    return ParametricAdaptiveReplayPlugin(
        mem_size=mem_size,
        batch_size=batch_size,
        temperature=temperature,
        n_clusters=n_clusters,
        vis_path=vis_path
    )

__all__ = ["ParametricAdaptiveReplayPlugin", "parametric_adaptive_replay"]

if __name__ == "__main__":
    print("Parametric Adaptive Replay Plugin for Avalanche")
    print("------------------------------------------")
    print("Import this module to use the ParametricAdaptiveReplayPlugin in your project.")
    print("Example usage: from parametric_replay import parametric_adaptive_replay")

