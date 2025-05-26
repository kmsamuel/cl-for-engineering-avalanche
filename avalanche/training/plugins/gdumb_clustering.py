import numpy as np
import torch
import copy
from typing import TYPE_CHECKING, Optional, List, Tuple
from torch.utils.data import TensorDataset, DataLoader

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
import os

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate

# Optional: Try to import TabPFN for embeddings
try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TabPFNRegressor = None
    TABPFN_AVAILABLE = False


class ClusteredBufferStoragePolicy:
    """
    Storage policy that keeps samples grouped by clusters.

    This is an extension of the buffer concept from Avalanche, 
    that adds clustering capabilities for more diverse sample selection.
    """
    
    def __init__(self, max_size: Optional[int] = 200, cluster_method: str = 'experience',
                 n_clusters: Optional[int] = None, adaptive_size: bool = True,
                 unlimited_buffer: bool = False):
        """
        Initialize the clustered buffer.
        
        Args:
            max_size: The maximum size of the buffer (None or -1 for unlimited)
            cluster_method: Method used for clustering ('experience', 'kmeans', 'tabpfn')
            n_clusters: Number of clusters to use (if None, determined by strategy)
            adaptive_size: If True, memory size is equally divided among clusters
            unlimited_buffer: If True, buffer size is unlimited (max_size is ignored)
        """
        self.max_size = max_size if not unlimited_buffer and max_size is not None and max_size > 0 else None
        self.cluster_method = cluster_method
        self.n_clusters = n_clusters
        self.adaptive_size = adaptive_size
        self.unlimited_buffer = unlimited_buffer
        
        # Buffer stores (x, y, embedding, cluster_id, experience_id)
        self.buffer = []
        self.current_experience_id = 0
        
        # Initialize TabPFN model if needed
        if cluster_method == 'tabpfn':
            if not TABPFN_AVAILABLE:
                raise ImportError("TabPFN must be installed for TabPFN clustering!")
            self.tabpfn_model = TabPFNRegressor()
    
    def compute_embeddings(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute embeddings depending on the cluster method."""
        if self.cluster_method == 'tabpfn':
            # Use TabPFN to compute embeddings
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
        else:
            # For experience-based or kmeans, just return inputs
            return X
    
    def assign_clusters(self, embeddings: np.ndarray) -> np.ndarray:
        """Assign cluster IDs to embeddings based on the clustering method."""
        if self.cluster_method == 'experience':
            # For experience-based clustering, use experience ID directly
            return np.array([eid for (_, _, _, _, eid) in self.buffer])
        else:
            # For KMeans or TabPFN, cluster the embeddings
            n_clusters = self._get_n_clusters(len(embeddings))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            return kmeans.fit_predict(embeddings)
    
    def _get_n_clusters(self, n_samples: int) -> int:
        """Determine number of clusters to use."""
        if self.n_clusters is not None:
            # User-specified number of clusters
            return min(self.n_clusters, n_samples)
        else:
            # Default: use number of experiences seen so far
            return min(max(1, self.current_experience_id + 1), n_samples)
    
    def update(self, strategy: "SupervisedTemplate", **kwargs) -> None:
        """
        Update buffer with data from current experience.
        """
        current_exp_id = strategy.experience.current_experience
        
        # Extract data
        X, y = self._extract_data_from_dataset(strategy.experience.dataset)
        if len(X) == 0:
            return
        
        X_new = np.stack(X)
        y_new = np.stack(y)
        
        # Compute embeddings for new data
        embeddings = self.compute_embeddings(X_new, y_new)
        
        # Add to buffer with placeholder cluster IDs
        new_entries = [(x, y_val, emb, -1, current_exp_id)
                     for x, y_val, emb in zip(X_new, y_new, embeddings)]
        self.buffer.extend(new_entries)
        
        # Update cluster assignments
        self._update_clusters()
        
        # Sample buffer to maintain max size (if not unlimited)
        if not self.unlimited_buffer and self.max_size is not None:
            self._sample_memory_buffer()
            
        self.current_experience_id = current_exp_id
    
    def _update_clusters(self) -> None:
        """Update cluster assignments for all samples in buffer."""
        if not self.buffer:
            return
        
        # Get all embeddings
        all_embeddings = np.stack([entry[2] for entry in self.buffer])
        
        # Assign clusters
        cluster_ids = self.assign_clusters(all_embeddings)
        
        # Update buffer entries with cluster IDs
        self.buffer = [(x, y, emb, cid, eid) 
                     for (cid, (x, y, emb, _, eid)) 
                     in zip(cluster_ids, self.buffer)]
    
    def _sample_memory_buffer(self) -> None:
        """Sample memory buffer with balanced representation from each cluster."""
        if not self.buffer or self.max_size is None or len(self.buffer) <= self.max_size:
            return
        
        # Group samples by cluster
        clusters = {}
        for entry in self.buffer:
            cid = entry[3]
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append(entry)
        
        # Sample from each cluster
        n_clusters = len(clusters)
        samples_per_cluster = self.max_size // n_clusters
        
        new_buffer = []
        for cid, samples in clusters.items():
            if len(samples) <= samples_per_cluster:
                new_buffer.extend(samples)
            else:
                sampled = random.sample(samples, samples_per_cluster)
                new_buffer.extend(sampled)
        
        # Handle remaining slots due to rounding
        remaining = self.max_size - len(new_buffer)
        if remaining > 0:
            # Use set of indices to track which buffer entries are already selected
            selected_indices = set()
            for entry in new_buffer:
                for i, buffer_entry in enumerate(self.buffer):
                    if self._are_entries_equal(entry, buffer_entry):
                        selected_indices.add(i)
                        break
            
            # Find all remaining indices
            all_indices = set(range(len(self.buffer)))
            remaining_indices = list(all_indices - selected_indices)
            
            if remaining_indices:
                # Sample from remaining indices
                extra_indices = random.sample(remaining_indices, min(remaining, len(remaining_indices)))
                for idx in extra_indices:
                    new_buffer.append(self.buffer[idx])
        
        self.buffer = new_buffer
    
    def _are_entries_equal(self, entry1, entry2):
        """Compare two buffer entries for equality, handling NumPy arrays properly."""
        # Compare experience IDs (scalar)
        if entry1[4] != entry2[4]:
            return False
        
        # Compare cluster IDs (scalar)
        if entry1[3] != entry2[3]:
            return False
        
        # Compare y values (scalar or array)
        y1, y2 = entry1[1], entry2[1]
        if isinstance(y1, np.ndarray) and isinstance(y2, np.ndarray):
            if not np.array_equal(y1, y2):
                return False
        elif y1 != y2:
            return False
        
        # Compare embeddings (arrays)
        emb1, emb2 = entry1[2], entry2[2]
        if not np.array_equal(emb1, emb2):
            return False
        
        # Compare x values (arrays)
        x1, x2 = entry1[0], entry2[0]
        if not np.array_equal(x1, x2):
            return False
        
        return True
    
    def _extract_data_from_dataset(self, dataset) -> Tuple[List, List]:
        """Extract features and targets from dataset."""
        X, y = [], []
        for i in range(len(dataset)):
            sample = dataset[i]
            if isinstance(sample, tuple) and len(sample) >= 2:
                X.append(sample[0].cpu().numpy() if isinstance(sample[0], torch.Tensor) else sample[0])
                y.append(sample[1])
        return X, y


    def get_avalanche_dataset(self, strategy: "SupervisedTemplate"):
        """Convert buffer to an Avalanche dataset for training."""
        if not self.buffer:
            return None
            
        # Extract data from buffer
        X_buf = np.stack([x for (x, _, _, _, _) in self.buffer])
        y_buf = np.stack([y for (_, y, _, _, _) in self.buffer])
        
        # Convert to tensors (keep on CPU initially)
        X_tensor = torch.tensor(X_buf, dtype=torch.float32)
        y_tensor = torch.tensor(y_buf, dtype=torch.float32)
        
        # Create the dataset on CPU
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Add collate_fn attribute to make compatible with Avalanche
        dataset.collate_fn = torch.utils.data._utils.collate.default_collate
        
        return dataset

    def get_training_sample(self, strategy: "SupervisedTemplate", training_budget: int):
        """
        Get a balanced sample for training with a specific budget.
        
        Args:
            strategy: The training strategy
            training_budget: Maximum number of samples to include
            
        Returns:
            Dataset with selected samples
        """
        if not self.buffer:
            return None
            
        # Group samples by cluster
        clusters = {}
        for entry in self.buffer:
            cid = entry[3]
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append(entry)
        
        # Calculate how many samples to take from each cluster
        n_clusters = len(clusters)
        samples_per_cluster = training_budget // n_clusters
        
        # Sample from each cluster
        selected_samples = []
        for cid, samples in clusters.items():
            if len(samples) <= samples_per_cluster:
                selected_samples.extend(samples)
            else:
                # Random sample from this cluster
                sampled = random.sample(samples, samples_per_cluster)
                selected_samples.extend(sampled)
        
        # Handle remaining slots due to rounding
        remaining = training_budget - len(selected_samples)
        if remaining > 0:
            # Use set of indices to track which buffer entries are already selected
            selected_indices = set()
            for entry in selected_samples:
                for i, buffer_entry in enumerate(self.buffer):
                    if self._are_entries_equal(entry, buffer_entry):
                        selected_indices.add(i)
                        break
            
            # Find all remaining indices
            all_indices = set(range(len(self.buffer)))
            remaining_indices = list(all_indices - selected_indices)
            
            if remaining_indices:
                # Sample from remaining indices
                extra_indices = random.sample(remaining_indices, min(remaining, len(remaining_indices)))
                for idx in extra_indices:
                    selected_samples.append(self.buffer[idx])
        
        # Convert to tensors
        if selected_samples:
            X_buf = np.stack([x for (x, _, _, _, _) in selected_samples])
            y_buf = np.stack([y for (_, y, _, _, _) in selected_samples])
            t_buf = np.stack([t for (_, _, t, _, _) in selected_samples])
            
            # Keep tensors on CPU - PyTorch DataLoader will handle device transfer
            X_tensor = torch.tensor(X_buf, dtype=torch.float32)
            y_tensor = torch.tensor(y_buf, dtype=torch.float32)
            t_tensor = torch.tensor(t_buf, dtype=torch.float32)
       
            dataset = TensorDataset(X_tensor, y_tensor, t_tensor)
            # Add collate_fn attribute to make compatible with Avalanche
            dataset.collate_fn = torch.utils.data._utils.collate.default_collate
            
            return dataset
        
        return None
    
    def visualize_clusters(self, save_path=None, title="Buffer Clusters Visualization"):
        """Visualize the buffer clustering using PCA projection."""
        if not self.buffer:
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


class ClusteredGDumbPlugin(SupervisedPlugin):
    """
    GDumb plugin with clustered buffer for improved sample selection.
    
    At each experience, the model is trained from scratch using samples selected
    from a buffer that balances across different clusters. The buffer is updated 
    at the start of each experience to incorporate new samples while maintaining 
    cluster balance.
    
    Three clustering methods are supported:
    - 'experience': Clusters by experience ID (original GDumb behavior)
    - 'kmeans': Clusters raw inputs using KMeans
    - 'tabpfn': Uses TabPFN embeddings before clustering with KMeans
    
    https://www.robots.ox.ac.uk/~tvg/publications/2020/gdumb.pdf
    """
    
    def __init__(self, 
                 mem_size: Optional[int] = 200, 
                 cluster_method: str = 'experience',
                 n_clusters: Optional[int] = None,
                 train_epochs: int = 1,
                 vis_path: Optional[str] = None,
                 unlimited_buffer: bool = False,
                 training_budget_per_experience: Optional[int] = None):
        """
        Initialize the ClusteredGDumbPlugin.
        
        Args:
            mem_size: Maximum number of samples in memory (None or -1 for unlimited)
            cluster_method: Method for clustering ('experience', 'kmeans', or 'tabpfn')
            n_clusters: Number of clusters (if None, uses number of experiences)
            train_epochs: Number of epochs to train on buffer after each experience
            vis_path: Path to save visualizations (None = no visualization)
            unlimited_buffer: If True, buffer size is unlimited (mem_size is ignored)
            training_budget_per_experience: If set, limits number of samples used for training
        """
        super().__init__()
        self.mem_size = mem_size
        self.cluster_method = cluster_method
        self.n_clusters = n_clusters
        self.train_epochs = train_epochs
        self.vis_path = vis_path
        self.unlimited_buffer = unlimited_buffer
        self.training_budget_per_experience = training_budget_per_experience
        
        # Storage policy
        self.storage_policy = ClusteredBufferStoragePolicy(
            max_size=self.mem_size,
            cluster_method=self.cluster_method,
            n_clusters=self.n_clusters,
            unlimited_buffer=self.unlimited_buffer
        )
        
        # Store initial model for resetting
        self.init_model = None
        
        # Make visualization directory if needed
        if self.vis_path:
            os.makedirs(self.vis_path, exist_ok=True)
    
    def before_train_dataset_adaptation(self, strategy: "SupervisedTemplate", **kwargs):
        """Reset model to initial state before training on next experience."""
        if self.init_model is None:
            self.init_model = copy.deepcopy(strategy.model)
        else:
            strategy.model = copy.deepcopy(self.init_model)
        
        # Apply any necessary model adaptations
        strategy.model_adaptation(strategy.model)
    
    def before_eval_dataset_adaptation(self, strategy: "SupervisedTemplate", **kwargs):
        """Ensure model is properly adapted before evaluation."""
        if self.init_model is not None:
            strategy.model_adaptation(strategy.model)
    
    def after_train_dataset_adaptation(self, strategy: "SupervisedTemplate", **kwargs):
        """Update buffer and set it as the training dataset."""
        # Update storage policy with new experience data
        self.storage_policy.update(strategy, **kwargs)
        
        # Get dataset for training
        if self.training_budget_per_experience is not None:
            # If training budget is specified, use a subset of the buffer
            buffer_dataset = self.storage_policy.get_training_sample(
                strategy, self.training_budget_per_experience
            )
            print(f"[GDumb] Training with budget of {self.training_budget_per_experience} samples")
        else:
            # Otherwise use the entire buffer
            buffer_dataset = self.storage_policy.get_avalanche_dataset(strategy)
        
        if buffer_dataset is not None:
            strategy.adapted_dataset = buffer_dataset
        
        # Visualize if needed
        if self.vis_path:
            self.storage_policy.visualize_clusters(
                save_path=os.path.join(
                    self.vis_path, 
                    f"experience_{strategy.experience.current_experience}_clusters.png"
                ),
                title=f"Clusters After Experience {strategy.experience.current_experience}"
            )
    
    def after_training_exp(self, strategy, **kwargs):
        """Print buffer statistics after training an experience."""
        print(f"[GDumb] Total buffer size: {len(self.storage_policy.buffer)} samples")
        if self.training_budget_per_experience is not None:
            print(f"[GDumb] Trained on {self.training_budget_per_experience} samples")


class ExperienceGDumbPlugin(ClusteredGDumbPlugin):
    """GDumb plugin that clusters samples by their originating experience."""
    
    def __init__(self, mem_size=200, train_epochs=1, vis_path=None, 
                 unlimited_buffer=False, training_budget_per_experience=None):
        super().__init__(
            mem_size=mem_size,
            cluster_method='experience',
            train_epochs=train_epochs,
            vis_path=vis_path,
            unlimited_buffer=unlimited_buffer,
            training_budget_per_experience=training_budget_per_experience
        )


class KMeansGDumbPlugin(ClusteredGDumbPlugin):
    """GDumb plugin that clusters samples using KMeans on raw inputs."""
    
    def __init__(self, mem_size=200, n_clusters=None, train_epochs=1, vis_path=None,
                 unlimited_buffer=False, training_budget_per_experience=None):
        super().__init__(
            mem_size=mem_size,
            cluster_method='kmeans',
            n_clusters=n_clusters,
            train_epochs=train_epochs,
            vis_path=vis_path,
            unlimited_buffer=unlimited_buffer,
            training_budget_per_experience=training_budget_per_experience
        )


class TabPFNGDumbPlugin(ClusteredGDumbPlugin):
    """GDumb plugin that clusters samples using KMeans on TabPFN embeddings."""
    
    def __init__(self, mem_size=200, n_clusters=None, train_epochs=1, vis_path=None,
                 unlimited_buffer=False, training_budget_per_experience=None):
        if not TABPFN_AVAILABLE:
            raise ImportError("TabPFN must be installed to use TabPFNGDumbPlugin!")
        
        super().__init__(
            mem_size=mem_size,
            cluster_method='tabpfn',
            n_clusters=n_clusters,
            train_epochs=train_epochs,
            vis_path=vis_path,
            unlimited_buffer=unlimited_buffer,
            training_budget_per_experience=training_budget_per_experience
        )


# Convenience function to create budgeted version for fair comparison
def create_budgeted_gdumb_plugin(
        cluster_method='experience',
        training_budget_per_experience=None,
        n_clusters=None, 
        train_epochs=1, 
        vis_path=None):
    """
    Create a GDumb plugin with unlimited buffer and fixed training budget.
    Useful for fair comparison with experience replay approaches.
    
    Args:
        cluster_method: Clustering method ('experience', 'kmeans', 'tabpfn')
        training_budget_per_experience: Number of samples to use in training
        n_clusters: Number of clusters to use
        train_epochs: Number of epochs to train
        vis_path: Path for visualizations
        
    Returns:
        ClusteredGDumbPlugin with unlimited buffer and controlled training budget
    """
    return ClusteredGDumbPlugin(
        mem_size=None,  # No limit
        cluster_method=cluster_method,
        n_clusters=n_clusters,
        train_epochs=train_epochs,
        vis_path=vis_path,
        unlimited_buffer=True,
        training_budget_per_experience=training_budget_per_experience
    )


# Export the plugins
__all__ = [
    "ClusteredGDumbPlugin",
    "ExperienceGDumbPlugin",
    "KMeansGDumbPlugin",
    "TabPFNGDumbPlugin",
    "create_budgeted_gdumb_plugin"
]

if __name__ == "__main__":
    print("Clustered GDumb Plugins Loaded.")