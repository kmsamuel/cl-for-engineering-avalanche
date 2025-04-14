"""
TabPFN Embedding Replay Strategy for Avalanche Continual Learning Framework

This module implements a novel continual learning replay strategy based on TabPFN embeddings
and feature space clustering to identify and prioritize samples for replay.
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from tabpfn import TabPFNRegressor
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Optional, Tuple, List, Dict, Union, Sequence

# Avalanche imports
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins.evaluation import EvaluationPlugin


class TabPFNReplayPlugin(SupervisedPlugin):
    """
    Avalanche plugin that implements a replay strategy based on TabPFN embeddings.
    
    This plugin extracts embeddings using TabPFN, clusters them to identify
    structure in the feature space, and selects samples for replay based on
    distribution shifts between experiences.
    """
    
    def __init__(self, 
                 mem_size: int = 200, 
                 n_clusters: int = 10, 
                 distance_threshold: float = 1.0, 
                 samples_per_cluster: int = 10):
        """
        Initialize the TabPFN embedding replay plugin.
        
        Args:
            mem_size: Maximum number of samples to keep in memory
            n_clusters: Number of clusters to use for KMeans
            distance_threshold: Minimum distance threshold to consider a cluster for replay
            samples_per_cluster: Maximum number of samples to select per cluster
        """
        super().__init__()
        self.mem_size = mem_size
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.samples_per_cluster = samples_per_cluster
        
        # Storage for previous experiences
        self.previous_experiences = []
        self.previous_embeddings = []
        self.previous_clusters = []
        self.previous_labels = []
        
        # TabPFN model for feature extraction
        self.model = TabPFNRegressor()
        
        # Memory buffer (to be compatible with Avalanche)
        self.buffer_x = None
        self.buffer_y = None
        self.buffer_tid = None
        
        # Flag to know if we're in training mode
        self.is_training = False
        
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
            X_tensor = X
        else:
            X_tensor = torch.tensor(X.astype(np.float32))
            
        if isinstance(y, torch.Tensor):
            y_tensor = y
        else:
            y_tensor = torch.tensor(y.astype(np.float32).reshape(-1, 1))
        
        try:
            # Fit the model 
            self.model.fit(X_tensor, y_tensor)
            
            # Call predict to ensure forward pass is completed
            self.model.predict(X_tensor, output_type='mean')
            
            # Access the embeddings from the model
            embeddings = self.model.model_.train_encoder_out.squeeze(1)
            return embeddings.detach().cpu().numpy()
        except Exception as e:
            print(f"Error extracting embeddings: {e}")
            # Return dummy embeddings in case of error
            return np.zeros((len(X_tensor), 10))
    
    def cluster_embeddings(self, embeddings):
        """
        Cluster the embeddings using KMeans.
        
        Args:
            embeddings: Embedding vectors to cluster
            
        Returns:
            Tuple of (cluster_labels, cluster_centers, kmeans_model)
        """
        # Ensure we don't try to create more clusters than samples
        n_clusters = min(self.n_clusters, len(embeddings))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        cluster_centers = kmeans.cluster_centers_
        return cluster_labels, cluster_centers, kmeans
    
    def store_experience(self, X, y, task_id=None):
        """
        Process and store a new experience.
        
        Args:
            X: Input features
            y: Target values
            task_id: Optional task identifier
        """
        # Extract embeddings
        embeddings = self.extract_embeddings(X, y)
        
        # Cluster the embeddings
        labels, centers, kmeans = self.cluster_embeddings(embeddings)
        
        # Store the experience
        self.previous_experiences.append((X, y, task_id))
        self.previous_embeddings.append(embeddings)
        self.previous_clusters.append(centers)
        self.previous_labels.append(labels)
        
        return embeddings, labels, centers
    
    def select_replay_samples(self, current_centers):
        """
        Select samples from previous experiences for replay based on
        cluster distances.
        
        Args:
            current_centers: Cluster centers from current data
            
        Returns:
            X_replay, y_replay, task_ids: Selected samples for replay
        """
        if not self.previous_experiences:
            return None, None, None
        
        all_replay_X = []
        all_replay_y = []
        all_replay_tid = []
        
        # For each previous experience
        for exp_idx, (prev_X, prev_y, prev_tid) in enumerate(self.previous_experiences):
            prev_centers = self.previous_clusters[exp_idx]
            prev_labels = self.previous_labels[exp_idx]
            
            # Compute distances between previous and current cluster centers
            distances = pairwise_distances(prev_centers, current_centers)
            
            # For each previous cluster, find the minimum distance to any current cluster
            min_distances = np.min(distances, axis=1)
            
            # Identify clusters that are distant from all current clusters
            distant_clusters = np.where(min_distances > self.distance_threshold)[0]
            
            replay_indices = []
            for cluster_idx in distant_clusters:
                # Find samples belonging to this cluster
                cluster_samples = np.where(prev_labels == cluster_idx)[0]
                
                # Select a subset of samples from this cluster
                if len(cluster_samples) > self.samples_per_cluster:
                    selected = np.random.choice(cluster_samples, self.samples_per_cluster, replace=False)
                else:
                    selected = cluster_samples
                    
                replay_indices.extend(selected)
            
            # Add selected samples to replay buffer
            if len(replay_indices) > 0:
                all_replay_X.append(prev_X[replay_indices])
                all_replay_y.append(prev_y[replay_indices])
                
                # Handle task IDs
                if prev_tid is not None:
                    if np.isscalar(prev_tid):
                        all_replay_tid.append(np.full(len(replay_indices), prev_tid))
                    else:
                        all_replay_tid.append(prev_tid[replay_indices])
                else:
                    all_replay_tid.append(np.full(len(replay_indices), exp_idx))
        
        # Combine samples from all previous experiences
        if all_replay_X:
            X_replay = np.vstack(all_replay_X)
            y_replay = np.concatenate(all_replay_y)
            task_ids = np.concatenate(all_replay_tid)
            
            # Limit to memory size if needed
            if len(X_replay) > self.mem_size:
                selected_indices = np.random.choice(len(X_replay), self.mem_size, replace=False)
                X_replay = X_replay[selected_indices]
                y_replay = y_replay[selected_indices]
                task_ids = task_ids[selected_indices]
                
            return X_replay, y_replay, task_ids
        else:
            return None, None, None
    
    def process_new_data(self, X, y, task_id=None):
        """
        Process new data and select samples for replay.
        
        Args:
            X: Input features of new data
            y: Target values of new data
            task_id: Optional task identifier
            
        Returns:
            X_replay, y_replay, task_ids: Selected samples for replay
        """
        # Extract embeddings for the new data
        current_embeddings = self.extract_embeddings(X, y)
        
        # Cluster the new data
        _, current_centers, _ = self.cluster_embeddings(current_embeddings)
        
        # Select replay samples based on cluster distances
        X_replay, y_replay, task_ids = self.select_replay_samples(current_centers)
        
        # Store the current experience
        self.store_experience(X, y, task_id)
        
        # Update buffer
        self.buffer_x = X_replay
        self.buffer_y = y_replay
        self.buffer_tid = task_ids
        
        return X_replay, y_replay, task_ids
    
    # Avalanche plugin methods
    def before_training_exp(self, strategy, **kwargs):
        """Called before training on an experience."""
        self.is_training = True
        
    def after_training_exp(self, strategy, **kwargs):
        """Called after training on an experience."""
        # Get the current experience data
        current_dataset = strategy.experience.dataset
        
        # Extract x and y from the dataset
        # This depends on how your dataset is structured
        X, y, task_id = self._extract_data_from_dataset(current_dataset)
        
        # Process the data and update the buffer
        self.process_new_data(X, y, task_id)
        
        self.is_training = False
    
    def before_training_iteration(self, strategy, **kwargs):
        """Called before a training iteration."""
        if not self.is_training or self.buffer_x is None or len(self.buffer_x) == 0:
            return
            
        # Get the current batch
        current_x, current_y = strategy.mb_x, strategy.mb_y
        
        # Select a subset of replay samples matching the batch size
        if len(self.buffer_x) > len(current_x):
            indices = np.random.choice(len(self.buffer_x), len(current_x), replace=False)
            replay_x = self.buffer_x[indices]
            replay_y = self.buffer_y[indices]
        else:
            replay_x = self.buffer_x
            replay_y = self.buffer_y
            
        # Convert to tensors if needed
        if not isinstance(replay_x, torch.Tensor):
            replay_x = torch.tensor(replay_x, device=current_x.device)
            replay_y = torch.tensor(replay_y, device=current_y.device)
        
        # Concatenate with current batch
        strategy.mb_x = torch.cat((current_x, replay_x))
        strategy.mb_y = torch.cat((current_y, replay_y))
    
    def _extract_data_from_dataset(self, dataset):
        """
        Extract data from an Avalanche dataset.
        
        Args:
            dataset: Avalanche dataset
            
        Returns:
            X, y, task_id: Extracted features, targets, and task ID
        """
        # This may need to be adapted based on your dataset structure
        X, y, task_labels = [], [], []
        
        # Handle both DatasetWithTargets and AvalancheDataset
        if hasattr(dataset, 'dataset'):
            # It's an AvalancheDataset
            if hasattr(dataset, '_x'):
                # Direct access to data
                X = dataset._x
                y = dataset._y
                task_labels = dataset.targets_task_labels
            else:
                # Iterate through the dataset
                for i in range(len(dataset)):
                    sample = dataset[i]
                    if isinstance(sample, tuple):
                        if len(sample) >= 2:
                            X.append(sample[0].numpy() if isinstance(sample[0], torch.Tensor) else sample[0])
                            y.append(sample[1])
                        if len(sample) >= 3:
                            task_labels.append(sample[2])
                X = np.array(X)
                y = np.array(y)
                if task_labels:
                    task_labels = np.array(task_labels)
                else:
                    task_labels = None
        else:
            # Try direct iteration
            for i in range(len(dataset)):
                sample = dataset[i]
                if isinstance(sample, tuple):
                    if len(sample) >= 2:
                        X.append(sample[0].numpy() if isinstance(sample[0], torch.Tensor) else sample[0])
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
    
    def visualize_embeddings(self, embeddings, labels, title="TabPFN Embeddings"):
        """
        Visualize embeddings using PCA.
        
        Args:
            embeddings: Embedding vectors
            labels: Corresponding labels or cluster assignments
            title: Plot title
        """
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Class/Cluster')
        plt.title(title)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.show()


# Factory function for easy integration with Avalanche
def tabpfn_embedding_replay(mem_size=200, n_clusters=10, distance_threshold=1.0, samples_per_cluster=10):
    """
    Factory function to create a TabPFNReplayPlugin instance.
    
    Args:
        mem_size: Maximum number of samples to keep in memory
        n_clusters: Number of clusters to use for KMeans
        distance_threshold: Minimum distance threshold to consider a cluster for replay
        samples_per_cluster: Maximum number of samples to select per cluster
        
    Returns:
        TabPFNReplayPlugin instance
    """
    return TabPFNReplayPlugin(
        mem_size=mem_size,
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        samples_per_cluster=samples_per_cluster
    )


# Example usage with Avalanche
def example_avalanche_usage():
    """Example showing how to use the plugin with Avalanche."""
    try:
        from avalanche.benchmarks.classic import SplitMNIST
        from avalanche.models import SimpleMLP
        from avalanche.training.supervised import Naive
        from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
        from avalanche.logging import InteractiveLogger
        from avalanche.training.plugins import EvaluationPlugin
        
        # Create a benchmark
        benchmark = SplitMNIST(n_experiences=5)
        
        # Model
        model = SimpleMLP(num_classes=10)
        
        # Create the evaluation plugin
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loggers=[InteractiveLogger()]
        )
        
        # Create the TabPFN replay plugin
        replay_plugin = tabpfn_embedding_replay(
            mem_size=500,
            n_clusters=15,
            distance_threshold=0.8,
            samples_per_cluster=10
        )
        
        # Create the strategy
        cl_strategy = Naive(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
            criterion=torch.nn.CrossEntropyLoss(),
            train_mb_size=128,
            eval_mb_size=128,
            train_epochs=1,
            eval_every=1,
            plugins=[replay_plugin],
            evaluator=eval_plugin
        )
        
        # Train on the benchmark
        print("Starting training...")
        for exp_idx, experience in enumerate(benchmark.train_stream):
            print(f"Experience {exp_idx}")
            cl_strategy.train(experience)
            cl_strategy.eval(benchmark.test_stream)
        
        print("Training completed!")
        
    except ImportError:
        print("To run this example, install Avalanche with: pip install avalanche-lib")
        

if __name__ == "__main__":
    print("TabPFN Replay Plugin for Avalanche")
    print("------------------------------------------")
    print("Import this module to use the TabPFNReplayPlugin in your project.")
    print("Example usage: from tabpfn_replay import tabpfn_embedding_replay")