
"""
TabPFN Embedding Replay Strategy for Avalanche Continual Learning Framework

This module implements a novel continual learning replay strategy based on TabPFN embeddings
and feature space clustering, integrated with Avalanche's buffer management.
The strategy follows a GDumb-inspired approach where representative samples are selected
based on the size of the current experience.
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import os
from tabpfn import TabPFNRegressor
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Optional, Tuple, List, Dict, Union, Sequence

# Avalanche imports
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.storage_policy import ExperienceBalancedBuffer


class TabPFNReplayPlugin(SupervisedPlugin):
    """
    Avalanche plugin that implements a replay strategy based on TabPFN embeddings,
    inspired by GDumb approach for continual learning.
    
    This plugin extracts embeddings using TabPFN, clusters them to identify
    structure in the feature space, and selects representative samples from all
    experiences for replay based on the size of the current experience.
    """
    
    def __init__(self, 
                 visualization_path: str = None,
                 adaptive_clusters: bool = True,
                 base_n_clusters: int = 10,
                 safety_cap: int = None):
        """
        Initialize the TabPFN embedding replay plugin.
        
        Args:
            visualization_path: Path to save visualizations (None = no visualizations)
            adaptive_clusters: Whether to adapt the number of clusters based on data size
            base_n_clusters: Base number of clusters (if adaptive_clusters=True, this is adjusted)
            safety_cap: Optional maximum limit on samples to keep (None = no limit)
        """
        super().__init__()
        self.visualization_path = visualization_path
        self.adaptive_clusters = adaptive_clusters
        self.base_n_clusters = base_n_clusters
        self.safety_cap = safety_cap
        
        # TabPFN model for feature extraction
        self.model = TabPFNRegressor()
        
        # Storage for experience embeddings and data
        self.experience_embeddings = {}
        self.stored_datasets = {}

        # Current experience tracking
        self.current_experience_id = 0
        
        # Storage policy for managing replay buffer
        self.storage_policy = ExperienceBalancedBuffer(
            max_size=1000000,  # Set very high as we'll manage size ourselves
            adaptive_size=True
        )
    
    def _determine_n_clusters(self, num_samples):
        """
        Determine the appropriate number of clusters based on data size.
        
        Args:
            num_samples: Number of samples in the data
            
        Returns:
            n_clusters: Number of clusters to use
        """
        if not self.adaptive_clusters:
            return self.base_n_clusters
        
        # Simple heuristic: sqrt(N/2) but capped
        n_clusters = int(np.sqrt(num_samples / 2))
        # Cap between 5 and 50 clusters
        return max(5, min(50, n_clusters))
    
    def extract_embeddings(self, X, y):
        """
        Extract embeddings from data using TabPFN.
        
        Args:
            X: Input features (numpy array or torch tensor)
            y: Target values (numpy array or torch tensor)
            
        Returns:
            Embeddings as numpy array
        """
        # Convert to appropriate format
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = X
            
        if isinstance(y, torch.Tensor):
            y_np = y.cpu().numpy().ravel()
        else:
            y_np = np.ravel(y) if hasattr(y, 'ravel') else y
        
        X_tensor = torch.tensor(X_np.astype(np.float32))
        y_tensor = torch.tensor(y_np.astype(np.float32))
        
        # Fit the model to extract embeddings
        try:
            self.model.fit(X_tensor, y_tensor)
            
            # Call predict to ensure forward pass is completed
            self.model.predict(X_tensor, output_type='mean')
            
            # Access the embeddings from the model
            embeddings = self.model.model_.train_encoder_out.squeeze(1)
            return embeddings.detach().cpu().numpy()
        except Exception as e:
            print(f"Error extracting embeddings: {e}")
            # Return a dummy embedding if there's an error
            return np.zeros((len(X_np), 10))
    
    def _extract_data_from_dataset(self, dataset):
        """
        Extract data from an Avalanche dataset.
        
        Args:
            dataset: Avalanche dataset
            
        Returns:
            X, y, task_id: Extracted features, targets, and task ID
        """
        X, y, task_labels = [], [], []
        
        # Handle dataset types
        if hasattr(dataset, "_flat_data_x"):
            # It's a flat Avalanche dataset with direct access to data
            X = dataset._flat_data_x
            y = dataset._flat_data_y
            if hasattr(dataset, "_flat_task_labels"):
                task_labels = dataset._flat_task_labels
            return X, y, task_labels
        
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
        if task_labels:
            task_labels = np.array(task_labels)
        else:
            task_labels = None
        
        return X, y, task_labels
    
    def _store_experience_embeddings(self, experience_id, embeddings, task_ids=None):
        """
        Store embeddings for an experience to compare later.
        
        Args:
            experience_id: ID of the current experience
            embeddings: TabPFN embeddings
            task_ids: Task IDs (if not provided, will use experience_id)
        """
        # If task_ids not provided, use experience_id for all samples
        if task_ids is None:
            task_ids = np.full(len(embeddings), experience_id)
        
        self.experience_embeddings[experience_id] = {
            'embeddings': embeddings,
            'task_ids': task_ids
        }
    
    def _select_representative_samples(self, embeddings, task_ids, n_clusters, n_samples):
        """
        Select representative samples using KMeans clustering with guaranteed sampling from each cluster.
        
        Args:
            embeddings: Embedding vectors
            task_ids: Task/experience IDs for each embedding
            n_clusters: Number of clusters to create
            n_samples: Total number of samples to select
            
        Returns:
            indices: Dictionary mapping experience_id to list of selected indices
        """
        # Ensure we don't create more clusters than samples
        n_clusters = min(n_clusters, len(embeddings))
        
        # Run KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Count samples in each cluster
        unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
        print(f"Cluster distribution: {dict(zip(unique_clusters, cluster_counts))}")
        
        # IMPORTANT: Ensure each cluster gets at least one sample
        samples_per_cluster = np.ones(n_clusters, dtype=int)
        remaining_samples = n_samples - n_clusters
        
        # If we have remaining samples, distribute proportionally
        if remaining_samples > 0:
            # Calculate proportions based on cluster size
            proportions = cluster_counts / np.sum(cluster_counts)
            # Distribute remaining samples
            additional_samples = np.round(proportions * remaining_samples).astype(int)
            samples_per_cluster += additional_samples
        
        # Adjust to ensure we get exactly n_samples total
        total_allocated = np.sum(samples_per_cluster)
        if total_allocated > n_samples:
            # If too many, reduce from largest allocations
            while total_allocated > n_samples:
                # Find largest allocation with more than 1 sample
                valid_indices = np.where(samples_per_cluster > 1)[0]
                if len(valid_indices) == 0:
                    break  # Can't reduce further while ensuring at least 1 per cluster
                idx = valid_indices[np.argmax(samples_per_cluster[valid_indices])]
                samples_per_cluster[idx] -= 1
                total_allocated -= 1
        elif total_allocated < n_samples:
            # If too few, add to smallest allocations
            while total_allocated < n_samples:
                idx = np.argmin(samples_per_cluster)
                samples_per_cluster[idx] += 1
                total_allocated += 1
        
        print(f"Samples per cluster: {dict(zip(unique_clusters, samples_per_cluster))}")
        
        # Select samples from each cluster
        all_selected_indices = []
        
        for i, cluster in enumerate(unique_clusters):
            # Get indices for this cluster
            cluster_indices = np.where(cluster_labels == cluster)[0]
            
            # Get number of samples to select for this cluster
            samples_to_select = min(samples_per_cluster[i], len(cluster_indices))
            
            if samples_to_select > 0:
                # Randomly select samples from this cluster
                selected = np.random.choice(cluster_indices, samples_to_select, replace=False)
                all_selected_indices.extend(selected)
        
        # Group selected indices by experience
        indices_by_experience = {}
        
        for idx in all_selected_indices:
            exp_id = task_ids[idx]
            if exp_id not in indices_by_experience:
                indices_by_experience[exp_id] = []
            
            # Calculate the local index within that experience
            local_idx = self._map_to_local_idx(exp_id, idx)
            indices_by_experience[exp_id].append(local_idx)
        
        return indices_by_experience
    
    def _map_to_local_idx(self, exp_id, global_idx):
        """
        Map a global index to a local index within an experience.
        This function needs to be customized based on how embeddings are stored.
        
        Args:
            exp_id: Experience ID
            global_idx: Global index in the combined embeddings
            
        Returns:
            local_idx: Local index within the experience
        """
        # This implementation assumes embeddings for each experience are stored separately
        # and global_idx is an index into the combined array of all embeddings
        
        # Keep track of the start index for each experience in the combined array
        start_indices = {}
        current_idx = 0
        
        # Calculate start indices
        for e_id in sorted(self.experience_embeddings.keys()):
            start_indices[e_id] = current_idx
            current_idx += len(self.experience_embeddings[e_id]['embeddings'])
        
        # Calculate local index
        if exp_id in start_indices:
            return global_idx - start_indices[exp_id]
        else:
            # Fallback - if we can't map properly, just return the global index
            # This shouldn't happen if the data structure is consistent
            print(f"Warning: Could not map global index {global_idx} to local index for experience {exp_id}")
            return global_idx

    def _map_local_to_global_indices(self, exp_id, local_indices):
        """Map local indices within an experience back to global indices."""
        # Calculate the offset for this experience
        offset = 0
        for eid in sorted(self.experience_embeddings.keys()):
            if eid == exp_id:
                break
            offset += len(self.experience_embeddings[eid]['embeddings'])
        
        # Map local indices to global
        return [offset + idx for idx in local_indices]
 
    def _collect_all_embeddings(self):
        """
        Collect all embeddings and task IDs from stored experiences.
        
        Returns:
            all_embeddings: Combined embeddings
            all_task_ids: Task IDs for each embedding
        """
        if not self.experience_embeddings:
            return None, None
        
        all_embeddings = []
        all_task_ids = []
        
        # Combine embeddings from all experiences
        for exp_id in sorted(self.experience_embeddings.keys()):
            exp_data = self.experience_embeddings[exp_id]
            all_embeddings.append(exp_data['embeddings'])
            all_task_ids.append(exp_data['task_ids'])
        
        # Stack/concatenate arrays
        all_embeddings = np.vstack(all_embeddings)
        all_task_ids = np.concatenate(all_task_ids)
        
        return all_embeddings, all_task_ids
 
    def visualize_clusters_with_selection(self, embeddings, cluster_labels, selected_indices=None, title="Clusters"):
        """
        Visualize clusters and optionally highlight selected samples.
        
        Args:
            embeddings: Embedding vectors
            cluster_labels: Cluster assignments
            selected_indices: Indices of selected samples (optional)
            title: Plot title
        """
        # Skip visualization if path not provided
        if self.visualization_path is None:
            return None
        
        # Create directory if needed
        os.makedirs(self.visualization_path, exist_ok=True)
        filepath = os.path.join(self.visualization_path, f"{title.replace(' ', '_')}.png")
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Count samples per cluster for analysis
        unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
        print(f"Cluster distribution: {dict(zip(unique_clusters, cluster_counts))}")
        
        # Count selected samples per cluster if provided
        if selected_indices is not None:
            selected_mask = np.zeros(len(embeddings), dtype=bool)
            selected_mask[selected_indices] = True
            
            selected_per_cluster = {}
            for cluster in unique_clusters:
                cluster_mask = cluster_labels == cluster
                n_selected = np.sum(selected_mask & cluster_mask)
                selected_per_cluster[cluster] = n_selected
            
            print(f"Selected per cluster: {selected_per_cluster}")
            
            # Calculate selection percentage
            selection_percentage = {c: (selected_per_cluster[c] / cluster_counts[i]) * 100 
                                for i, c in enumerate(unique_clusters)}
            print(f"Selection percentage per cluster: {selection_percentage}")
        
        plt.figure(figsize=(12, 10))
        
        # Plot all samples with cluster coloring
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                            c=cluster_labels, cmap='viridis', alpha=0.5, s=30)
        
        # If selected indices provided, highlight them
        if selected_indices is not None:
            plt.scatter(embeddings_2d[selected_indices, 0], embeddings_2d[selected_indices, 1],
                    c=cluster_labels[selected_indices], cmap='viridis', 
                    alpha=1.0, s=100, edgecolors='black', linewidths=1)
        
        # Add cluster count annotation
        for i, (cluster, count) in enumerate(zip(unique_clusters, cluster_counts)):
            # Find center of this cluster
            cluster_mask = cluster_labels == cluster
            centroid_x = np.mean(embeddings_2d[cluster_mask, 0])
            centroid_y = np.mean(embeddings_2d[cluster_mask, 1])
            
            # Add annotation
            if selected_indices is not None:
                n_selected = selected_per_cluster[cluster]
                plt.annotate(f"C{cluster}: {count}\nSel: {n_selected}", 
                            (centroid_x, centroid_y), fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
            else:
                plt.annotate(f"C{cluster}: {count}", 
                            (centroid_x, centroid_y), fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        
        plt.colorbar(scatter, label='Cluster')
        plt.title(title)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        
        # Add legend if selected indices provided
        if selected_indices is not None:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                    markersize=8, alpha=0.5, label='All samples'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                    markersize=10, alpha=1.0, markeredgecolor='black', label='Selected')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
        
        # Save and close
        plt.savefig(filepath)
        plt.close()
        
        print(f"Visualization saved to: {filepath}")
        return filepath
    
    def visualize_embeddings(self, embeddings, labels, title="TabPFN Embeddings"):
        """
        Visualize embeddings using PCA.
        
        Args:
            embeddings: Embedding vectors
            labels: Corresponding labels or cluster assignments
            title: Plot title
            
        Returns:
            filepath: Path to saved visualization file
        """
        # Skip visualization if path not provided
        if self.visualization_path is None:
            return None
        
        # Create directory if needed
        os.makedirs(self.visualization_path, exist_ok=True)
        filepath = os.path.join(self.visualization_path, f"{title.replace(' ', '_')}.png")
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Class/Cluster')
        plt.title(title)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        
        # Save and close
        plt.savefig(filepath)
        plt.close()
        
        print(f"Visualization saved to: {filepath}")
        return filepath
    
    # Avalanche plugin methods
    def before_training_exp(self, strategy, **kwargs):
        """Called before training on an experience."""
        # Get current experience ID
        self.current_experience_id = strategy.experience.current_experience
        print(f"Processing experience {self.current_experience_id}")
        
        # If this is the first experience, just train normally
        if self.current_experience_id == 0:
            return
        
        # Not the first experience - prepare custom dataset from previous embeddings
        # Collect all previous embeddings
        all_embeddings, all_task_ids = self._collect_all_embeddings()
        
        if all_embeddings is None or len(all_embeddings) == 0:
            print("Warning: No previous embeddings available")
            return
        
        # Get the current experience data to determine sample count
        current_dataset = strategy.experience.dataset
        X_current, _, _ = self._extract_data_from_dataset(current_dataset)
        current_experience_size = len(X_current)
        
        print(f"Current experience size: {current_experience_size}")
        
        # Determine number of clusters based on data size
        n_clusters = self._determine_n_clusters(len(all_embeddings))
        print(f"Using {n_clusters} clusters for sample selection")
        
        # Select representative samples based on current experience size
        n_samples = current_experience_size
        
        # Select representative samples
        indices_by_exp = self._select_representative_samples(
            all_embeddings, all_task_ids, n_clusters, n_samples)
        
        # Create a custom dataset from selected samples
        combined_x = []
        combined_y = []
        combined_task_ids = []
        
        for exp_id, indices in indices_by_exp.items():
            if exp_id < self.current_experience_id and exp_id in self.stored_datasets:
                # Get the dataset
                dataset = self.stored_datasets[exp_id]
                X, y, task_id = self._extract_data_from_dataset(dataset)
                
                # Select samples
                selected_X = X[indices]
                selected_y = y[indices]
                
                # Create task IDs if not provided
                if task_id is None:
                    selected_task_ids = np.full(len(indices), exp_id)
                else:
                    selected_task_ids = task_id[indices]
                
                # Add to combined dataset
                combined_x.append(selected_X)
                combined_y.append(selected_y)
                combined_task_ids.append(selected_task_ids)
        
        # If we have selected samples, create a new dataset
        if combined_x:
            # Combine all selected samples
            combined_x = np.vstack(combined_x) if len(combined_x[0].shape) > 1 else np.concatenate(combined_x)
            combined_y = np.concatenate(combined_y)
            combined_task_ids = np.concatenate(combined_task_ids)
            
            print(f"Created combined dataset with {len(combined_x)} samples")
            
            # Create torch dataset
            import torch
            from torch.utils.data import TensorDataset, DataLoader
            
            # Convert to torch tensors
            X_tensor = torch.tensor(combined_x)
            y_tensor = torch.tensor(combined_y)
            tasks_tensor = torch.tensor(combined_task_ids.astype(np.int64))
            
            # Create dataset and dataloader
            custom_dataset = TensorDataset(X_tensor, y_tensor, tasks_tensor)
            
            # Replace strategy's dataset and dataloader
            strategy.adapted_dataset = custom_dataset
            strategy.dataloader = DataLoader(
                custom_dataset,
                batch_size=strategy.train_mb_size,
                shuffle=True
            )
            
            print(f"Replaced strategy's dataset and dataloader with selected samples")
        
    # def before_training_iteration(self, strategy, **kwargs):
    #     """Called before each training iteration."""
    #     # Get the current batch
    #     batch_x = strategy.mb_x
    #     batch_y = strategy.mb_y
        
    #     # Basic batch information
    #     print(f"\n--- BATCH INFO ---")
    #     print(f"Batch size: {len(batch_x)}")
    #     print(f"Labels in batch: {np.unique(batch_y.cpu().numpy())}")
        
    #     # Try to determine which experience each sample came from
    #     if hasattr(strategy, 'adapted_dataset') and hasattr(strategy.adapted_dataset, 'indices'):
    #         # If using ReplayDataLoader with AvalancheSubset
    #         indices = strategy.adapted_dataset.indices
    #         print(f"Dataset indices: {indices[:20]}...")  # Show first 20 to avoid clutter
        
    #     # If using your dataset storage
    #     if hasattr(self, 'stored_datasets'):
    #         # Try to match samples with stored datasets
    #         match_found = False
    #         for exp_id, dataset in self.stored_datasets.items():
    #             if hasattr(dataset, '_flat_data_x'):
    #                 # Check if any samples in the batch match this dataset
    #                 # Note: This is a simple check and might not work for all data types
    #                 for i, sample in enumerate(batch_x[:5]):  # Check first 5 samples
    #                     sample_np = sample.cpu().numpy().flatten()
    #                     for j, stored_sample in enumerate(dataset._flat_data_x[:100]):  # Check first 100 stored
    #                         stored_np = stored_sample.flatten() if hasattr(stored_sample, 'flatten') else stored_sample
    #                         if np.array_equal(sample_np, stored_np):
    #                             print(f"Sample {i} in batch matches experience {exp_id}, sample {j}")
    #                             match_found = True
            
    #         if not match_found:
    #             print("No matches found between batch and stored samples")
        
    #     # Check if using a ReplayDataLoader
    #     if hasattr(strategy, 'dataloader') and 'ReplayDataLoader' in str(type(strategy.dataloader)):
    #         print("Using ReplayDataLoader")
    #         if hasattr(strategy.dataloader, 'dataset_memory'):
    #             mem_len = len(strategy.dataloader.dataset_memory)
    #             print(f"Memory dataset size: {mem_len}")
            
    def after_training_exp(self, strategy, **kwargs):
        """Called after training on an experience."""
        # Get the current experience data
        dataset = strategy.experience.dataset
        
        # Extract data
        X, y, task_id = self._extract_data_from_dataset(dataset)
        
        if len(X) == 0:
            print("Warning: No data extracted from dataset")
            return
        
        # Extract embeddings
        print(f"Extracting embeddings for experience {self.current_experience_id}")
        embeddings = self.extract_embeddings(X, y)
        
        # If no task ID was provided, use the current experience ID
        if task_id is None:
            task_id = np.full(len(X), self.current_experience_id)
        
        # Store embeddings for this experience
        self._store_experience_embeddings(self.current_experience_id, embeddings, task_id)
        
        # Visualize embeddings if visualization is enabled
        if self.visualization_path is not None:
            self.visualize_embeddings(
                embeddings,
                task_id,
                title=f"Experience_{self.current_experience_id}_Embeddings"
            )
        
        # If this is the first experience, store all samples
        if self.current_experience_id == 0:
            print(f"First experience: Storing embeddings for all {len(X)} samples")
            
            # For the first experience, we don't need to do anything special with the buffer
            # Just store the embeddings
        
        # # Clear buffer for the next experience (we'll rebuild it in before_training_exp)
        if hasattr(self.storage_policy, 'reset_current_buffer'):
            self.storage_policy.reset_current_buffer()
        elif hasattr(self.storage_policy, '_reset_buffer'):
            self.storage_policy._reset_buffer()
        else:
            # Recreate the storage policy if no reset method exists
            self.storage_policy = ExperienceBalancedBuffer(
                max_size=1000000,  # Set very high as we'll manage size ourselves
                adaptive_size=True
            )

        self.stored_datasets[self.current_experience_id] = dataset
        
        # Update experience counter
        self.current_experience_id += 1

# Factory function for easy integration with Avalanche
def tabpfn_embedding_replay(visualization_path=None, adaptive_clusters=True, base_n_clusters=10, safety_cap=None):
    """
    Factory function to create a TabPFNReplayPlugin instance.
    
    Args:
        visualization_path: Path to save visualizations (None = no visualizations)
        adaptive_clusters: Whether to adapt the number of clusters based on data size
        base_n_clusters: Base number of clusters
        safety_cap: Optional maximum limit on samples to keep (None = no limit)
        
    Returns:
        TabPFNReplayPlugin instance
    """
    return TabPFNReplayPlugin(
        visualization_path=visualization_path,
        adaptive_clusters=adaptive_clusters,
        base_n_clusters=base_n_clusters,
        safety_cap=safety_cap
    )

__all__ = ["TabPFNReplayPlugin", "tabpfn_embedding_replay"]

if __name__ == "__main__":
    print("TabPFN Replay Plugin for Avalanche")
    print("------------------------------------------")
    print("Import this module to use the TabPFNReplayPlugin in your project.")
    print("Example usage: from tabpfn_replay import tabpfn_embedding_replay")