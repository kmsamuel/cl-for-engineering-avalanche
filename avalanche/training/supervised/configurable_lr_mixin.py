"""
Configurable Learning Rate Mixin for your Avalanche repository.

Add this file to your avalanche repo at:
avalanche/training/strategies/configurable_lr_mixin.py
"""

from typing import Dict, Any, Optional, Union
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class ConfigurableLRMixin:
    """
    Mixin class to add configurable learning rate scheduling to any strategy.
    
    This replaces the hardcoded learning rate adjustments in your strategy classes
    with a configurable system that can be customized per benchmark.
    """
    
    def __init__(self, lr_config: Optional[Dict[str, Any]] = None, *args, **kwargs):
        """
        Initialize the configurable LR mixin.
        
        :param lr_config: Dictionary containing LR scheduling configuration
        """
        super().__init__(*args, **kwargs)
        self.lr_config = lr_config or {}
        self.current_epoch = 0
        self._cosine_scheduler = None
        
        # Setup cosine scheduler if needed
        if self.lr_config.get('type') == 'cosine':
            self._setup_cosine_scheduler()
    
    def _setup_cosine_scheduler(self):
        """Setup cosine annealing scheduler."""
        if hasattr(self, 'optimizer'):
            self._cosine_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.lr_config.get('T_0', 5),
                T_mult=self.lr_config.get('T_mult', 2),
                eta_min=self.lr_config.get('eta_min', 5e-6)
            )
    
    def apply_lr_schedule(self, epoch: int):
        """
        Apply learning rate schedule based on configuration.
        
        :param epoch: Current epoch number
        """
        if not self.lr_config:
            return
            
        schedule_type = self.lr_config.get('type', 'manual')
        
        if schedule_type == 'manual':
            self._manual_lr_adjustment(epoch)
        elif schedule_type == 'cosine':
            if self._cosine_scheduler is not None:
                self._cosine_scheduler.step()
        # Add other scheduler types as needed
    
    def _manual_lr_adjustment(self, epoch: int):
        """
        Manual learning rate adjustment based on milestones.
        
        This implements the exact same logic as your original hardcoded versions,
        but makes it configurable via the lr_config parameter.
        """
        initial_lr = self.lr_config.get('initial_lr', 0.001)
        cycle_length = self.lr_config.get('cycle_length', 150)
        milestones = self.lr_config.get('milestones', [50, 100])
        gamma = self.lr_config.get('gamma', 0.1)
        
        epoch_in_cycle = epoch % cycle_length
        current_lr = initial_lr
        
        # Apply milestone-based reduction (same logic as your original code)
        for milestone in sorted(milestones, reverse=True):
            if epoch_in_cycle >= milestone:
                reduction_count = len([m for m in milestones if epoch_in_cycle >= m])
                current_lr = initial_lr * (gamma ** reduction_count)
                break
        
        # Update optimizer learning rate
        if hasattr(self, 'optimizer'):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
    
    def configurable_training_epoch(self, **kwargs):
        """
        Training epoch with configurable LR scheduling.
        
        Call this from your strategy's training_epoch method.
        """
        # Apply learning rate schedule
        self.apply_lr_schedule(self.current_epoch)
        
        # Display current learning rate
        if hasattr(self, 'optimizer'):
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'**Current LR**: {current_lr:.6f}')
        
        # Increment epoch counter
        self.current_epoch += 1
    
    def reset_lr_scheduler(self):
        """Reset learning rate scheduler state."""
        self.current_epoch = 0
        if self._cosine_scheduler is not None:
            self._setup_cosine_scheduler()


# Pre-defined configurations for your benchmarks
class BenchmarkLRConfigs:
    """Pre-defined LR configurations for different benchmarks."""
    
    @staticmethod
    def shapenet_cars():
        """LR config for ShapeNet Cars benchmark."""
        return {
            'type': 'manual',
            'initial_lr': 0.001,
            'milestones': [50, 100],
            'gamma': 0.1,
            'cycle_length': 150
        }
    
    @staticmethod
    def drivaernet_pointclouds():
        """LR config for DrivAerNet Point Clouds benchmark."""
        return {
            'type': 'manual',
            'initial_lr': 0.001,
            'milestones': [50, 100, 150],
            'gamma': 0.1,
            'cycle_length': 200
        }
    
    @staticmethod
    def drivaernet_parametric():
        """LR config for DrivAerNet Parametric benchmark."""
        return {
            'type': 'manual',
            'initial_lr': 0.001,
            'milestones': [50, 100],
            'gamma': 0.1,
            'cycle_length': 150
        }
    
    @staticmethod
    def shipd_pointclouds():
        """LR config for SHIPD Point Clouds benchmark."""
        return {
            'type': 'cosine',
            'initial_lr': 0.001,
            'T_0': 5,
            'T_mult': 2,
            'eta_min': 5e-6
        }
    
    @staticmethod
    def shipd_parametric():
        """LR config for SHIPD Parametric benchmark."""
        return {
            'type': 'cosine',
            'initial_lr': 0.01,  # Higher LR for SHIPD parametric
            'T_0': 5,
            'T_mult': 2,
            'eta_min': 5e-6
        }
    
    @staticmethod
    def raadl():
        """LR config for RAADL benchmark."""
        return {
            'type': 'cosine',
            'initial_lr': 0.001,
            'T_0': 10,
            'T_mult': 1,
            'eta_min': 1e-6
        }
    
    @staticmethod
    def get_config(benchmark_name: str):
        """Get configuration by benchmark name."""
        configs = {
            'shapenet_cars': BenchmarkLRConfigs.shapenet_cars,
            'drivaernet_pointclouds': BenchmarkLRConfigs.drivaernet_pointclouds,
            'drivaernet_parametric': BenchmarkLRConfigs.drivaernet_parametric,
            'shipd_pointclouds': BenchmarkLRConfigs.shipd_pointclouds,
            'shipd_parametric': BenchmarkLRConfigs.shipd_parametric,
            'raadl': BenchmarkLRConfigs.raadl,
        }
        
        if benchmark_name not in configs:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        return configs[benchmark_name]()