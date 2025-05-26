import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

from typing import Optional, TYPE_CHECKING

from packaging.version import parse
import torch

from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    ExperienceBalancedBuffer,
)
from avalanche.training.plugins import ReplayPlugin
if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate

# ReplayTracker - Simple plugin to track which samples are replayed
class ReplayTracker(SupervisedPlugin):
    def __init__(self, visualization_path=None):
        super().__init__()
        self.visualization_path = visualization_path or './'
        self.replayed_indices = set()  # Track which memory samples were replayed
        
    def before_training_exp(self, strategy, **kwargs):
        """Reset tracking for new experience"""
        self.replayed_indices = set()
        
    def before_training_iteration(self, strategy, **kwargs):
        """Track memory indices with debugging"""
        if hasattr(strategy, 'dataloader') and isinstance(strategy.dataloader, ReplayDataLoader):
            # Debug: print relevant attributes
            print("ReplayDataLoader attributes:")
            for attr in dir(strategy.dataloader):
                if not attr.startswith('_') or attr == '_latest_mem_indices':
                    print(f"  {attr}")
                    
            # Check mb_it (mini-batch iterator)
            if hasattr(strategy, 'mb_it'):
                print("Mini-batch iterator attributes:")
                for attr in dir(strategy.mb_it):
                    if not attr.startswith('_') or attr == '_latest_mem_indices':
                        print(f"  {attr}")
            
            # Check for the various possible locations
            if hasattr(strategy.mb_it, 'buffer_idxs'):
                print(f"Found {len(strategy.mb_it.buffer_idxs)} indices in mb_it.buffer_idxs")
                self.replayed_indices.update(strategy.mb_it.buffer_idxs)
            elif hasattr(strategy.dataloader, 'buffer_idxs'):
                print(f"Found {len(strategy.dataloader.buffer_idxs)} indices in dataloader.buffer_idxs")
                self.replayed_indices.update(strategy.dataloader.buffer_idxs)
            elif hasattr(strategy.dataloader, '_latest_mem_indices'):
                print(f"Found {len(strategy.dataloader._latest_mem_indices)} indices in _latest_mem_indices")
                self.replayed_indices.update(strategy.dataloader._latest_mem_indices)
            else:
                print("Could not find memory indices")
    
    def after_training_exp(self, strategy, **kwargs):
        """Create visualization after training on an experience"""
        # Find the replay plugin to access the buffer
        replay_plugin = None
        for plugin in strategy.plugins:
            if plugin.__class__.__name__ == 'ReplayPlugin':
                replay_plugin = plugin
                break
                
        if replay_plugin is None or not hasattr(replay_plugin, 'storage_policy'):
            return
            
        # Access the buffer
        buffer = replay_plugin.storage_policy.buffer
        if buffer is None or len(buffer) == 0:
            return
            
        # Get PCA embeddings of all buffer samples
        X = self._extract_features(buffer)
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(X)
        
        # Create colors: gray for buffer, red for replayed
        colors = ['red' if i in self.replayed_indices else 'gray' 
                 for i in range(len(buffer))]
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.7)
        plt.title("Buffer Samples: Replayed (red) vs. Not Replayed (gray)")
        plt.savefig(f"{self.visualization_path}/replay_visualization.png")
        plt.close()
        
        print(f"Visualized {len(self.replayed_indices)}/{len(buffer)} replayed samples")
        
    def _extract_features(self, dataset):
        """Extract features from dataset for PCA visualization"""
        # Implement based on your dataset format
        features = []
        for i in range(len(dataset)):
            sample = dataset[i]
            x = sample[0] if isinstance(sample, tuple) else sample
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
            features.append(x.flatten())
        return np.array(features)