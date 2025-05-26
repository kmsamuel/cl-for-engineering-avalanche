# from typing import Callable, Optional, List, Union
from typing import Callable, Optional, Sequence, List, Union, Dict, Any

import torch

from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts

from avalanche.benchmarks.utils.utils import concat_datasets
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.templates.strategy_mixin_protocol import CriterionType
# Import the new mixin
from .configurable_lr_mixin import ConfigurableLRMixin, BenchmarkLRConfigs


# class Cumulative(SupervisedTemplate):
#     """Cumulative training strategy.

#     At each experience, train model with data from all previous experiences
#         and current experience.
#     """

#     def __init__(
#         self,
#         *,
#         model: Module,
#         optimizer: Optimizer,
#         criterion: CriterionType,
#         train_mb_size: int = 1,
#         train_epochs: int = 1,
#         eval_mb_size: Optional[int] = None,
#         device: Union[str, torch.device] = "cpu",
#         plugins: Optional[List[SupervisedPlugin]] = None,
#         evaluator: Union[
#             EvaluationPlugin, Callable[[], EvaluationPlugin]
#         ] = default_evaluator,
#         eval_every=-1,
#         scheduler_type=None,
#         **kwargs
#     ):
#         """Init.

#         :param model: The model.
#         :param optimizer: The optimizer to use.
#         :param criterion: The loss criterion to use.
#         :param train_mb_size: The train minibatch size. Defaults to 1.
#         :param train_epochs: The number of training epochs. Defaults to 1.
#         :param eval_mb_size: The eval minibatch size. Defaults to 1.
#         :param device: The device to use. Defaults to None (cpu).
#         :param plugins: Plugins to be added. Defaults to None.
#         :param evaluator: (optional) instance of EvaluationPlugin for logging
#             and metric computations.
#         :param eval_every: the frequency of the calls to `eval` inside the
#             training loop. -1 disables the evaluation. 0 means `eval` is called
#             only at the end of the learning experience. Values >0 mean that
#             `eval` is called every `eval_every` epochs and at the end of the
#             learning experience.
#         """

#         super().__init__(
#             model=model,
#             optimizer=optimizer,
#             criterion=criterion,
#             train_mb_size=train_mb_size,
#             train_epochs=train_epochs,
#             eval_mb_size=eval_mb_size,
#             device=device,
#             plugins=plugins,
#             evaluator=evaluator,
#             eval_every=eval_every,
#             **kwargs
#         )

#         self.dataset = None  # cumulative dataset
#         # self.current_epoch = 0  # Track the current epoch
        
#         self.scheduler_type = scheduler_type
#         # Set up appropriate scheduler based on type
#         ## RAADL-PC
#         # if self.scheduler_type == 'cosine':
#         #     self.scheduler = CosineAnnealingWarmRestarts(
#         #         optimizer,
#         #         T_0=10,  # First cycle of 10 epochs (RAADL)
#         #         # T_0=20,  # First cycle of 10 epochs
#         #         T_mult=1,  # Keep cycle length constant
#         #         eta_min=1e-6
#         #     )
#         ## SHIPD-PC
#         if self.scheduler_type == 'cosine':
#             self.scheduler = CosineAnnealingWarmRestarts(
#                 optimizer,
#                 T_0=5,  # Shorter first cycle to allow multiple restarts
#                 T_mult=2,  # Double the cycle length after each restart
#                 eta_min=5e-6  # Slightly higher minimum learning rate
#             )
#         else: # Use manual adjustment
#             self.scheduler = None
        
#         self.current_epoch = 0  # Track the current epoch

#     def train_dataset_adaptation(self, **kwargs):
#         """
#         Concatenates all the previous experiences.
#         """
#         exp = self.experience
#         assert exp is not None
#         if self.dataset is None:
#             self.dataset = exp.dataset
#         else:
#             self.dataset = concat_datasets([self.dataset, exp.dataset])
#         self.adapted_dataset = self.dataset
            
#     # def training_epoch(self, **kwargs):
#     #     """Training epoch with adaptive learning rate."""
#     #     # Adjust the learning rate at the beginning of the epoch
#     #     effective_epoch = self.current_epoch
#     #     self.adjust_learning_rate(self.optimizer, effective_epoch)
        
#     #     current_lr = self.optimizer.param_groups[0]['lr']
#     #     print('**opt LR**', current_lr)
        
#     #     # Proceed with the standard training loop
#     #     super().training_epoch(**kwargs)
        
#     #     self.current_epoch += 1
        
#     def adjust_learning_rate(self, optimizer, epoch):
#         # initial_lr = 0.01 # shipd
#         initial_lr = 0.001 #everything else i think

#         # epoch_in_cycle = epoch % 150
#         # # More precise milestone checking, for metamaterials
#         # # if epoch_in_cycle >= 350:
#         # #     current_lr = initial_lr * (0.1 ** 5)
#         # # elif epoch_in_cycle >= 250:
#         # #     current_lr = initial_lr * (0.1 ** 4)
#         # if epoch_in_cycle >= 135:
#         #     current_lr = initial_lr * (0.1 ** 3)
#         # elif epoch_in_cycle >= 90:
#         #     current_lr = initial_lr * (0.1 ** 2)
#         # elif epoch_in_cycle >= 40:
#         #     current_lr = initial_lr * 0.1
#         # else:
#         #     current_lr = initial_lr
        
#         # # For shapenet, drivaernet, shipd(point clouds)
#         epoch_in_cycle = epoch % 200 # 200 for drivaernet point clouds
#         # epoch_in_cycle = epoch % 150 #  for drivaernet parametric
        
#         if epoch_in_cycle >= 150: # 150 for drivaernet PC, take out for drivaernet par
#             current_lr = initial_lr * (0.1 ** 3)
#         elif epoch_in_cycle >= 100: 
#         # if epoch_in_cycle >= 100: # DN parametric
#             current_lr = initial_lr * (0.1 ** 2)
#         elif epoch_in_cycle >= 50:
#             current_lr = initial_lr * 0.1
#         else:
#             current_lr = initial_lr        
        
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = current_lr
    
#     def training_epoch(self, **kwargs):
#         """Training epoch."""

#         if self.scheduler is None:
#             # Use manual adjustment if no scheduler type specified
#             effective_epoch = self.current_epoch
#             if self.scheduler_type == 'manual':
#                 self.adjust_learning_rate(self.optimizer, effective_epoch)
        
#         # Display current learning rate
#         current_lr = self.optimizer.param_groups[0]['lr']
#         print(f'**Current LR**: {current_lr:.6f}')
        
#         # Proceed with the standard training loop
#         super().training_epoch(**kwargs)
        
#         # Step the scheduler after the epoch (if using a scheduler)
#         if self.scheduler is not None and isinstance(self.scheduler, CosineAnnealingWarmRestarts):
#             self.scheduler.step()
                    
#         self.current_epoch += 1     
 
class Cumulative(ConfigurableLRMixin, SupervisedTemplate):
    """
    Cumulative training strategy with configurable learning rate scheduling.

    At each experience, train model with data from all previous experiences
    and current experience.
    """

    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        lr_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            lr_config=lr_config,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **kwargs
        )

        self.dataset = None  # cumulative dataset

    def train_dataset_adaptation(self, **kwargs):
        """
        Concatenates all the previous experiences.
        """
        exp = self.experience
        assert exp is not None
        
        if self.dataset is None:
            self.dataset = exp.dataset
        else:
            self.dataset = concat_datasets([self.dataset, exp.dataset])
        
        self.adapted_dataset = self.dataset
            
    def training_epoch(self, **kwargs):
        """Training epoch with configurable learning rate scheduling."""
        self.configurable_training_epoch()
        super().training_epoch(**kwargs)


# Helper function for your benchmark scripts
def create_strategy_with_lr_config(
    strategy_name: str,
    benchmark_name: str,
    model: Module,
    optimizer: Optimizer,
    criterion: CriterionType,
    **strategy_kwargs
):
    """
    Helper function to create a strategy with appropriate LR config for the benchmark.
    
    :param strategy_name: Name of the strategy ('naive', 'cumulative', etc.)
    :param benchmark_name: Name of the benchmark (for LR config)
    :param model: The model
    :param optimizer: The optimizer
    :param criterion: Loss criterion
    :param strategy_kwargs: Additional strategy arguments
    :return: Configured strategy instance
    """
    # Get the appropriate LR config for this benchmark
    try:
        lr_config = BenchmarkLRConfigs.get_config(benchmark_name)
    except ValueError:
        print(f"Warning: No LR config found for benchmark '{benchmark_name}', using default")
        lr_config = BenchmarkLRConfigs.shapenet_cars()  # Default fallback
    
    # Create the strategy with LR config
    if strategy_name == 'naive':
        return Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            lr_config=lr_config,
            **strategy_kwargs
        )
    elif strategy_name == 'cumulative':
        return Cumulative(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            lr_config=lr_config,
            **strategy_kwargs
        )
    elif strategy_name == 'generative':
        return GenerativeStrategy(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            lr_config=lr_config,
            **strategy_kwargs
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

       
__all__ = ["Cumulative",
           "create_strategy_with_lr_config",
           "BenchmarkLRConfigs"
                ]
