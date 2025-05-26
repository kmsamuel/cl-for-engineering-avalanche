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
# from NSMAB.Models.LDSModeller import LDSModeller
# from NSMAB.Policies.Policies import NSMAB_Policy_Compositional
# from NSMAB.Samplers import BoltzmannSampling

# from tabpfn import TabPFNRegressor
# import torch.nn.functional as F
# import random
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# class TabPFNAdaptiveReplayPlugin(SupervisedPlugin):
#     def __init__(self, mem_size=500, batch_size=32, temperature=1.0, n_clusters=4, vis_path=None):
#         super().__init__()
#         self.mem_size = mem_size
#         self.batch_size = batch_size
#         self.temperature = temperature
#         self.n_clusters = n_clusters
#         self.vis_path = vis_path

#         self.buffer = []  # Stores (x, y, embedding, cluster_id)
#         self.kmeans = None
#         self.tabpfn_model = TabPFNRegressor()
#         self.criterion = torch.nn.MSELoss(reduction='none')

#         modeller_ctor = lambda _: MovingAverage(alpha=0.7, assumed_increase_in_mean=10.0)
#         sampling_strategy = BoltzmannSampling(softmax_temperature=self.temperature)
#         self.bandit_policy = NSMAB_Policy_Compositional(modeller_ctor, sampling_strategy)

#     def extract_embeddings(self, X, y):
#         X_tensor = torch.tensor(X.astype(np.float32))
#         y_tensor = torch.tensor(y.astype(np.float32).ravel())
#         try:
#             self.tabpfn_model.fit(X_tensor, y_tensor)
#             self.tabpfn_model.predict(X_tensor, output_type='mean')
#             embeddings = self.tabpfn_model.model_.train_encoder_out.squeeze(1)
#             return embeddings.detach().cpu().numpy()
#         except Exception as e:
#             print(f"Embedding extraction failed: {e}")
#             return np.zeros((len(X), 10))

#     def assign_clusters(self, embeddings):
#         self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
#         return self.kmeans.fit_predict(embeddings)
    
#     def visualize_embeddings(self, embeddings, labels, title="TabPFN Embeddings"):
#         """
#         Visualize embeddings using PCA.
        
#         Args:
#             embeddings: Embedding vectors
#             labels: Corresponding labels or cluster assignments
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
        
#         # Reduce to 2D for visualization
#         pca = PCA(n_components=2)
#         embeddings_2d = pca.fit_transform(embeddings)
        
#         plt.figure(figsize=(10, 8))
#         scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
#         plt.colorbar(scatter, label='Class/Cluster')
#         plt.title(title)
#         plt.xlabel('PCA Component 1')
#         plt.ylabel('PCA Component 2')
        
#         # Save and close
#         plt.savefig(filepath)
#         plt.close()
        
#         print(f"Visualization saved to: {filepath}")
#         return filepath
    
#     # def visualize_clusters(self, embeddings, cluster_ids, selected_clusters, title='Selected Clusters'):
#     #     pca = PCA(n_components=2)
#     #     reduced = pca.fit_transform(embeddings)
#     #     plt.figure(figsize=(10, 8))
#     #     colors = ["#cccccc"] * len(embeddings)
#     #     for i, cid in enumerate(cluster_ids):
#     #         if cid in selected_clusters:
#     #             colors[i] = "red"
#     #     plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, s=15, alpha=0.8)
#     #     plt.title(title)
                
#     #     os.makedirs(self.vis_path, exist_ok=True)
#     #     filepath = os.path.join(self.vis_path, f"{title.replace(' ', '_')}.png")
        
#     #     plt.savefig(filepath)
#     #     plt.close()
#     #     print(f"[BanditReplayPlugin] Saved cluster visualization to {self.vis_path}")
        
#     def visualize_clusters(self, embeddings, cluster_ids, selected_clusters, title='Selected Clusters'):
#         """
#         Visualize selected clusters with consistent style matching the embeddings visualization.
        
#         Args:
#             embeddings: Embedding vectors
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
#         pca = PCA(n_components=2)
#         embeddings_2d = pca.fit_transform(embeddings)
        
#         # Create a figure with the same size and style as the embeddings visualization
#         plt.figure(figsize=(8, 8))
        
#         # Color points based on selection (red for selected, gray for non-selected)
#         colors = np.array(["#cccccc"] * len(embeddings))
#         for i, cid in enumerate(cluster_ids):
#             if cid in selected_clusters:
#                 colors[i] = "red"
        
#         # Use the exact same scatter plot parameters for consistency
#         plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.7)
        
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
        
#         self.current_experience_id = strategy.experience.current_experience
#         print(f"[BanditReplayPlugin] Processing experience {self.current_experience_id}")

#         if self.current_experience_id == 0:
#             print("[BanditReplayPlugin] Experience 0 — using default Avalanche dataloader, no replay.")
#             return
        
#         # Reset bandit policy to prevent arm accumulation
#         modeller_ctor = lambda _: MovingAverage(alpha=0.7, assumed_increase_in_mean=10.0)
#         sampling_strategy = BoltzmannSampling(softmax_temperature=self.temperature)
#         self.bandit_policy = NSMAB_Policy_Compositional(modeller_ctor, sampling_strategy)
#         self.bandit_policy.add_clusters(self.n_clusters)
 
            
#         if not self.buffer:
#             print("[BanditReplayPlugin] Warning: Memory buffer is empty.")
#             return

#         # Filter buffer to previous experiences only
#         past_buffer = [(x, y, emb, cid, eid) for (x, y, emb, cid, eid) in self.buffer if eid < self.current_experience_id]
#         print(f"[BanditReplayPlugin] Buffer size: {len(self.buffer)}")
#         print(f"[BanditReplayPlugin] Past buffer size: {len(past_buffer)}")
#         if not past_buffer:
#             print("[BanditReplayPlugin] Warning: No valid memory samples for replay.")
#             return

#         # Group samples by cluster ID
#         cluster_to_samples = {i: [] for i in range(self.n_clusters)}
#         for x, y, emb, cid, eid in past_buffer:
#             cluster_to_samples.setdefault(cid, []).append((x, y, emb))

#         available_cluster_ids = list(range(self.n_clusters))
#         print(f"[BanditReplayPlugin] Available cluster IDs: {available_cluster_ids}")
#         k = min(self.batch_size, len(available_cluster_ids))
#         selected_cluster_idxs = self.bandit_policy.suggest_clusters_to_sample_from(k)
#         print(f"[BanditReplayPlugin] Selected cluster indices from bandit: {selected_cluster_idxs}")

#         # Map cluster index (from policy) back to actual cluster ID
#         if max(selected_cluster_idxs, default=-1) >= len(available_cluster_ids):
#             print(f"[BanditReplayPlugin] Warning: Bandit returned invalid cluster index. Available: {len(available_cluster_ids)}, Got: {selected_cluster_idxs}")
#             return
#         selected_clusters = [available_cluster_ids[i] for i in selected_cluster_idxs]
        
#         selected_cluster_ids = self.bandit_policy.suggest_clusters_to_sample_from(k)
#         replay_emb = [emb for (_, _, emb, cid, eid) in self.buffer if eid != self.current_experience_id]
#         replay_cids = [cid for (_, _, _, cid, eid) in self.buffer if eid != self.current_experience_id]
        
#         print(f"[DEBUG] Experience_{self.current_experience_id}_Selected_Clusters: {len(replay_emb)} points")

#         if replay_emb and replay_cids:
#             self.visualize_clusters(
#                 embeddings=np.array(replay_emb),
#                 cluster_ids=replay_cids,
#                 selected_clusters=selected_cluster_ids,
#                 title=f"Experience_{self.current_experience_id}_Selected_Clusters"
#             )
            
#         selected_x, selected_y, selected_ids = [], [], []
#         for cid in selected_clusters:
#             samples = cluster_to_samples.get(cid, [])
#             if not samples:
#                 print(f"[BanditReplayPlugin] Skipping cluster {cid} — no samples available.")
#                 continue
#             num_samples = min(len(samples), max(1, 550 // k))
#             chosen = random.sample(samples, num_samples)
#             for x, y, _ in chosen:
#                 selected_x.append(x)
#                 selected_y.append(y)
#                 selected_ids.append(cid)

#         selected_x = np.stack(selected_x)
#         selected_y = np.stack(selected_y)

#         # # Log replay labels
#         # unique_labels, label_counts = np.unique(selected_y, return_counts=True)
#         # label_stats = dict(zip(unique_labels, label_counts))
#         # print(f"[BanditReplayPlugin] Replay label distribution: {label_stats}")

#         # Create dataloader for selected memory samples
#         tensor_x = torch.tensor(selected_x)
#         tensor_y = torch.tensor(selected_y)
#         task_ids = torch.tensor(selected_ids)

#         dataset = TensorDataset(tensor_x, tensor_y, task_ids)
#         # Combine current experience with selected memory samples
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
#         # strategy.dataloader = strategy._make_train_dataloader(full_dataset)
#         strategy.dataloader = DataLoader(
#                 full_dataset,
#                 batch_size=64,
#                 shuffle=True
#             )
#         # strategy.dataloader = DataLoader(
#         #     full_dataset,
#         #     batch_size=32,
#         #     shuffle=True,
#         #     num_workers=4,
#         #     pin_memory=True,
#         #     drop_last=True,
#         #     persistent_workers=True
#         # )

#         # Log full batch label distribution
#         # full_labels, full_counts = np.unique(combined_y, return_counts=True)
#         # full_stats = dict(zip(full_labels, full_counts))
#         # print(f"[BanditReplayPlugin] Combined batch label distribution: {full_stats}")
#         # print(f"[BanditReplayPlugin] Combined batch labels: {combined_y.tolist()}")

#         print(f"[BanditReplayPlugin] Injected {len(selected_x)} replay samples from selected clusters into strategy.dataloader")


#     def before_training_iteration(self, strategy, **kwargs):
#         # print("\n[BanditReplayPlugin] --- BATCH INFO ---")
#         # print(f"Batch size: {len(strategy.mb_x)}")
#         # print(f"Batch labels: {np.unique(strategy.mb_y.cpu().numpy())}")
        
#         self.current_experience_id = strategy.experience.current_experience
#         if self.current_experience_id == 0:
#             return
        
#         with torch.no_grad():
#             outputs = strategy.model(strategy.mb_x)
#             losses = self.criterion(outputs.squeeze(), strategy.mb_y.squeeze())
#             strategy.mb_loss = losses.detach().cpu().numpy()

#     def after_training_exp(self, strategy, **kwargs):
#         self.bandit_policy.add_clusters(self.n_clusters)
#         dataset = strategy.experience.dataset
#         X, y = self._extract_data_from_dataset(dataset)
#         embeddings = self.extract_embeddings(X, y)
#         cluster_ids = self.assign_clusters(embeddings)

#         sample_size = max(1, int(0.2 * len(X)))
#         idxs = np.random.choice(len(X), sample_size, replace=False)
#         X_sample = X[idxs]
#         y_sample = y[idxs]
#         emb_sample = embeddings[idxs]
#         cid_sample = cluster_ids[idxs]

#         print(f"[BanditReplayPlugin] Using {sample_size} samples for loss estimation.")

#         with torch.no_grad():
#             model = strategy.model.to(strategy.device)
#             x_tensor = torch.tensor(X).to(strategy.device)
#             y_tensor = torch.tensor(y).to(strategy.device)
#             preds = model(x_tensor)
#             losses = self.criterion(preds.squeeze(), y_tensor.squeeze()).cpu().numpy()

#         current_exp_id = strategy.experience.current_experience
#         print(f"[DEBUG] Experience_{current_exp_id}_Embeddings: {len(embeddings)} points")

#         for x, y_val, emb, cid, loss in zip(X_sample, y_sample, emb_sample, cid_sample, losses):
#             self.buffer.append((x, y_val, emb, cid, current_exp_id))
#             self.bandit_policy.log_observations({cid: np.array([loss])})

#         if len(self.buffer) > self.mem_size:
#             self.buffer = self.buffer[-self.mem_size:]
            
#         # Visualize embeddings if visualization is enabled
#         if self.vis_path is not None:
#             self.visualize_embeddings(
#                 embeddings,
#                 cluster_ids,
#                 title=f"Experience_{self.current_experience_id}_Embeddings"
#             )            
#     # def _extract_data_from_dataset(self, dataset):
#     #     X, y = [], []
#     #     for i in range(len(dataset)):
#     #         sample = dataset[i]
#     #         if isinstance(sample, tuple) and len(sample) >= 2:
#     #             X.append(sample[0].cpu().numpy() if isinstance(sample[0], torch.Tensor) else sample[0])
#     #             y.append(sample[1].cpu().numpy() if isinstance(sample[1], torch.Tensor) else sample[1])
#     #     return np.array(X), np.array(y)

#     def _extract_data_from_dataset(self, dataset):
#         """
#         Extract data from an Avalanche dataset.
        
#         Args:
#             dataset: Avalanche dataset
            
#         Returns:
#             X, y, task_id: Extracted features, targets, and task ID
#         """
#         X, y, task_labels = [], [], []
        
#         # Handle dataset types
#         if hasattr(dataset, "_flat_data_x"):
#             # It's a flat Avalanche dataset with direct access to data
#             X = dataset._flat_data_x
#             y = dataset._flat_data_y
#             if hasattr(dataset, "_flat_task_labels"):
#                 task_labels = dataset._flat_task_labels
#             return X, y, task_labels
        
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
#         if task_labels:
#             task_labels = np.array(task_labels)
#         else:
#             task_labels = None
        
#         return X, y

# __all__ = ["TabPFNAdaptiveReplayPlugin"]

# if __name__ == "__main__":
#     print("TabPFN Adaptive Replay Plugin for Avalanche")
#     print("------------------------------------------")
#     print("Import this module to use the TabPFNAdaptiveReplayPlugin in your project.")
#     print("Example usage: from tabpfn_replay import tabpfn_embedding_replay")

#############################################################################################################################################################################

import numpy as np
import torch
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from avalanche.training.plugins import SupervisedPlugin
from torch.utils.data import DataLoader, TensorDataset
import sys, os
from tabpfn import TabPFNRegressor
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
from collections import Counter

# Add BanditCL to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
banditcl_path = os.path.join(current_dir, "BanditCL")
sys.path.append(banditcl_path)

from NSMAB.Models.MovingAverage import MovingAverage
from NSMAB.Models.LDSModeller import LDSModeller
from NSMAB.Policies.Policies import NSMAB_Policy_Compositional
from NSMAB.Samplers import BoltzmannSampling

class TabPFNAdaptiveReplayPlugin(SupervisedPlugin):
    def __init__(self, mem_size=500, batch_size=32, temperature=1.0, n_clusters=4, vis_path=None):
        super().__init__()
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.temperature = temperature
        self.n_clusters = n_clusters
        self.vis_path = vis_path

        self.buffer = []  # Stores (x, y, embedding, cluster_id, experience_id)
        self.kmeans = None
        self.tabpfn_model = TabPFNRegressor()
        self.criterion = torch.nn.MSELoss(reduction='none')

        modeller_ctor = lambda _: MovingAverage(alpha=0.7, assumed_increase_in_mean=10.0)
        sampling_strategy = BoltzmannSampling(softmax_temperature=self.temperature)
        self.bandit_policy = NSMAB_Policy_Compositional(modeller_ctor, sampling_strategy)
        self.bandit_policy.add_clusters(self.n_clusters)

    def extract_embeddings(self, X, y):
        X_tensor = torch.tensor(X.astype(np.float32))
        y_tensor = torch.tensor(y.astype(np.float32).ravel())
        try:
            self.tabpfn_model.fit(X_tensor, y_tensor)
            self.tabpfn_model.predict(X_tensor, output_type='mean')
            embeddings = self.tabpfn_model.model_.train_encoder_out.squeeze(1)
            return embeddings.detach().cpu().numpy()
        except Exception as e:
            print(f"Embedding extraction failed: {e}")
            return np.zeros((len(X), 10))

    def assign_clusters(self, embeddings):
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        return self.kmeans.fit_predict(embeddings)

    def visualize_clusters(self, embeddings, cluster_ids, selected_sample_indices, title='Selected Samples'):
        """
        Visualize all embeddings with selected samples in red, all others in gray.
        
        Args:
            embeddings: Full set of embeddings
            cluster_ids: Cluster assignment for each sample
            selected_sample_indices: Indices of samples selected for replay
            title: Plot title
        """
        if self.vis_path is None:
            return None

        os.makedirs(self.vis_path, exist_ok=True)
        filepath = os.path.join(self.vis_path, f"{title.replace(' ', '_')}.png")

        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        plt.figure(figsize=(8, 8))
        colors = np.array(["#cccccc"] * len(embeddings))
        for i in selected_sample_indices:
            colors[i] = "red"

        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.7)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Selected Samples'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#cccccc', markersize=10, label='Non-selected')
        ]
        plt.legend(handles=legend_elements)

        selected_count = len(selected_sample_indices)
        total_count = len(embeddings)
        percentage = 100 * selected_count / total_count if total_count > 0 else 0

        plt.figtext(0.5, 0.01,
                    f"Selected samples: {selected_count}/{total_count} ({percentage:.1f}%)",
                    ha='center', fontsize=10)

        plt.title(title)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.savefig(filepath)
        plt.close()

        print(f"[BanditReplayPlugin] Selected sample visualization saved to: {filepath}")
        return filepath
    
    
    def visualize_embeddings(self, embeddings, labels, title="TabPFN Embeddings"):
        """
        Visualize embeddings using PCA, colored by cluster ID.
        """
        if self.vis_path is None:
            return None

        os.makedirs(self.vis_path, exist_ok=True)
        filepath = os.path.join(self.vis_path, f"{title.replace(' ', '_')}.png")

        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.7
        )
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(title)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.savefig(filepath)
        plt.close()

        print(f"[BanditReplayPlugin] Embedding visualization saved to: {filepath}")
        return filepath

    def before_training_exp(self, strategy, **kwargs):
        self.current_experience_id = strategy.experience.current_experience
        print(f"[BanditReplayPlugin] Processing experience {self.current_experience_id}")
        
        if self.current_experience_id == 0:
            print("[BanditReplayPlugin] Experience 0 — using default Avalanche dataloader, no replay.")
            return

        # Get current experience data
        current_x, current_y = self._extract_data_from_dataset(strategy.experience.dataset)

        if not self.buffer:
            print("[BanditReplayPlugin] Warning: Memory buffer is empty.")
            return

        # Get past buffer samples only
        past_buffer = [(x, y, emb, cid, eid) for (x, y, emb, cid, eid) in self.buffer if eid < self.current_experience_id]
        print(f"[BanditReplayPlugin] Buffer size: {len(self.buffer)}")
        print(f"[BanditReplayPlugin] Past buffer size: {len(past_buffer)}")

        if not past_buffer:
            print("[BanditReplayPlugin] Warning: No valid memory samples for replay.")
            return
        
        embedding_list = [emb for (_, _, emb, _, _) in past_buffer]
        cluster_ids_buffer = [cid for (_, _, _, cid, _) in past_buffer]
        
        if self.vis_path is not None:
            self.visualize_embeddings(embedding_list, cluster_ids_buffer,
                                      title=f"Experience_{self.current_experience_id}_Past_Buffer_Clusters")
            
        # Group buffer samples by cluster
        cluster_to_samples = {i: [] for i in range(self.n_clusters)}
        for x, y, emb, cid, eid in past_buffer:
            cluster_to_samples[cid].append((x, y, eid))

        # --- Bandit selects clusters (with possible repeats) ---
        k = min(self.batch_size, self.n_clusters)
        bandit_selected = self.bandit_policy.suggest_clusters_to_sample_from(k)

        # # --- Force full cluster coverage ---
        selected_clusters = set(bandit_selected)
        # for c in range(self.n_clusters):
        #     selected_clusters.add(c)
        # selected_clusters = list(selected_clusters)
        print(f"[BanditReplayPlugin] Selected clusters: {selected_clusters}")

        # --- Count how many times each cluster was selected ---
        cluster_counts = Counter(bandit_selected)
        # Assign weight 1 to clusters not selected by bandit
        for cid in selected_clusters:
            if cid not in cluster_counts:
                cluster_counts[cid] = 1  # ensures non-selected clusters still contribute

        # --- Compute sample budget per cluster ---
        total_weight = sum(cluster_counts.values())
        samples_per_cluster = {
            cid: max(1, int((cluster_counts[cid] / total_weight) * self.mem_size))
            for cid in selected_clusters
        }
        
        print("Current experience:", self.current_experience_id)
        print("Full buffer size:", len(self.buffer))
        print("Past buffer size:", len(past_buffer))
        print("All buffer experience IDs:", [eid for (_, _, _, _, eid) in self.buffer])

        # --- Sample from each cluster accordingly ---
        selected_x, selected_y, selected_ids, selected_eids = [], [], [], []
        for cid in selected_clusters:
            samples = cluster_to_samples.get(cid, [])
            if not samples:
                continue
            num_samples = min(len(samples), samples_per_cluster[cid])
            chosen = random.sample(samples, num_samples)
            for x, y, eid in chosen:
                selected_x.append(x)
                selected_y.append(y)
                selected_ids.append(cid)
                selected_eids.append(eid)

        print("All selected cluster IDs:", selected_ids)
        print("All selected experience IDs:", selected_eids)

        # === Map selected_x to past_buffer indices for visualization ===
        past_index_map = {id(x): i for i, (x, _, _, _, _) in enumerate(past_buffer)}
        selected_sample_indices = []
        for x in selected_x:
            idx = past_index_map.get(id(x), None)
            if idx is not None:
                selected_sample_indices.append(idx)

        # Convert to arrays
        selected_x = np.stack(selected_x)
        selected_y = np.stack(selected_y)

        # # Bandit selects clusters
        # k = min(self.batch_size, self.n_clusters)
        # selected_cluster_idxs = self.bandit_policy.suggest_clusters_to_sample_from(k)
        # selected_clusters = [i for i in selected_cluster_idxs]
        

        # Sample from selected clusters
        # selected_x, selected_y, selected_ids, selected_eids = [], [], [], []
        # for cid in selected_clusters:
        #     samples = cluster_to_samples.get(cid, [])
        #     if not samples:
        #         continue
        #     num_samples = min(len(samples), max(1, 550 // k))
        #     chosen = random.sample(samples, num_samples)
        #     for x, y, eid in chosen:
        #         selected_x.append(x)
        #         selected_y.append(y)
        #         selected_ids.append(cid)
        #         selected_eids.append(eid)
        

        # # Convert to arrays
        # selected_x = np.stack(selected_x)
        # selected_y = np.stack(selected_y)
        
        current_x = np.array(current_x)
        current_y = np.array(current_y)
        
        replayed_embeddings = self.extract_embeddings(selected_x, selected_y)

        # Combine for visualization and training
        combined_x = np.concatenate([current_x, selected_x])
        combined_y = np.concatenate([current_y, selected_y])
        combined_ids = np.concatenate([np.full(len(current_x), self.current_experience_id), np.array(selected_ids)])

        # Embed with current encoder
        combined_embeddings = self.extract_embeddings(combined_x, combined_y)
        # selected_indices = list(range(len(current_x), len(current_x) + len(selected_x)))

        if self.vis_path is not None:
            self.visualize_clusters(
                embeddings=np.stack(embedding_list),
                cluster_ids=cluster_ids_buffer,
                selected_sample_indices=selected_sample_indices,
                title=f"Experience_{self.current_experience_id}_Selected_Samples"
            )
                    

        # Build training dataset
        tensor_x = torch.tensor(combined_x)
        tensor_y = torch.tensor(combined_y)
        task_ids = torch.tensor(combined_ids)

        full_dataset = TensorDataset(tensor_x, tensor_y, task_ids)
        strategy.adapted_dataset = full_dataset
        strategy.dataloader = DataLoader(full_dataset, batch_size=64, shuffle=True)

        print(f"[BanditReplayPlugin] Injected {len(selected_x)} replay samples from selected clusters into strategy.dataloader")

    def before_training_iteration(self, strategy, **kwargs):
        self.current_experience_id = strategy.experience.current_experience
        if self.current_experience_id == 0:
            return
        with torch.no_grad():
            outputs = strategy.model(strategy.mb_x)
            losses = self.criterion(outputs.squeeze(), strategy.mb_y.squeeze())
            strategy.mb_loss = losses.detach().cpu().numpy()

    def after_training_exp(self, strategy, **kwargs):
        """
        After training on the current experience:
        - Add all current experience samples to buffer
        - Trim buffer to keep self.mem_size samples evenly across experiences
        - Extract TabPFN embeddings over the full buffer
        - Cluster embeddings and assign new cluster IDs
        - Log loss per updated cluster to bandit
        """
        # === Step 1: Add full current experience ===
        X, y = self._extract_data_from_dataset(strategy.experience.dataset)
        current_exp_id = strategy.experience.current_experience
        X_new = np.stack(X)
        y_new = np.stack(y)
        emb_new = self.extract_embeddings(X_new, y_new)
        new_entries = [(x, y_val, emb, -1, current_exp_id)
                    for x, y_val, emb in zip(X_new, y_new, emb_new)]

        # Extend buffer with new entries
        self.buffer.extend(new_entries)

        # === Step 2: Trim and rebalance buffer by experience ===
        all_exp_ids = sorted(set(eid for (_, _, _, _, eid) in self.buffer))
        num_exps = len(all_exp_ids)
        samples_per_exp = self.mem_size // num_exps

        # Trim per experience
        rebalanced = []
        for eid in all_exp_ids:
            exp_samples = [entry for entry in self.buffer if entry[4] == eid]
            if len(exp_samples) > samples_per_exp:
                exp_samples = random.sample(exp_samples, samples_per_exp)
            rebalanced.extend(exp_samples)
        self.buffer = rebalanced

        # === Step 3: Cluster over all embeddings in buffer ===
        all_embs = np.stack([entry[2] for entry in self.buffer])  # emb = 3rd entry
        cluster_ids = self.assign_clusters(all_embs)

        # === Step 4: Rebuild buffer with new cluster assignments ===
        self.buffer = [
            (x, y, emb, cid, eid)
            for (cid, (x, y, emb, _, eid)) in zip(cluster_ids, self.buffer)
        ]

        # === Step 5: Compute losses across all buffer samples ===
        X_buf = np.stack([x for (x, _, _, _, _) in self.buffer])
        y_buf = np.stack([y for (_, y, _, _, _) in self.buffer])
        cid_buf = [cid for (_, _, _, cid, _) in self.buffer]

        with torch.no_grad():
            model = strategy.model.to(strategy.device)
            preds = model(torch.tensor(X_buf).to(strategy.device))
            losses = self.criterion(preds.squeeze(), torch.tensor(y_buf.squeeze()).to(strategy.device)).cpu().numpy()
            
        # === Step 6: Log updated losses to bandit using current cluster IDs ===
        cluster_logs = {}
        for cid, loss in zip(cid_buf, losses):
            cluster_logs.setdefault(cid, []).append(loss)

        self.bandit_policy.log_observations({cid: np.array(l) for cid, l in cluster_logs.items()})
        self.bandit_policy.next_timestep()

        print(f"[BanditReplayPlugin] Buffer updated with embeddings and reclustered ({len(self.buffer)} total).")

    def _extract_data_from_dataset(self, dataset):
        X, y = [], []
        for i in range(len(dataset)):
            sample = dataset[i]
            if isinstance(sample, tuple) and len(sample) >= 2:
                X.append(sample[0].cpu().numpy() if isinstance(sample[0], torch.Tensor) else sample[0])
                y.append(sample[1])
        return np.array(X), np.array(y)

__all__ = ["TabPFNAdaptiveReplayPlugin"]

if __name__ == "__main__":
    print("TabPFN Adaptive Replay Plugin for Avalanche")
    print("------------------------------------------")
    print("Import this module to use the TabPFNAdaptiveReplayPlugin in your project.")
    print("Example usage: from tabpfn_replay import tabpfn_embedding_replay")