from typing import Optional, TYPE_CHECKING

from packaging.version import parse
import torch
import hashlib

from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    ExperienceBalancedBuffer,
)

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


# def get_model_param_hash(model: torch.nn.Module) -> str:
#     """Returns an MD5 hash of the model's parameters."""
#     with torch.no_grad():
#         flat_params = torch.cat([p.data.view(-1).cpu() for p in model.parameters()])
#     return hashlib.md5(flat_params.numpy().tobytes()).hexdigest()
# def batch_hash(batch_x):
#     return hashlib.md5(batch_x.cpu().numpy().tobytes()).hexdigest()
# def get_optimizer_hash(optimizer):
#     state_tensors = []
#     for state in optimizer.state.values():
#         for v in state.values():
#             if isinstance(v, torch.Tensor):
#                 state_tensors.append(v.view(-1).cpu())
#     if not state_tensors:
#         return None
#     all_data = torch.cat(state_tensors).numpy().tobytes()
#     return hashlib.sha256(all_data).hexdigest()
# def hash_model_params(model):
#     param_bytes = b''.join([p.detach().cpu().numpy().tobytes() for p in model.parameters()])
#     return hashlib.md5(param_bytes).hexdigest()

class ReplayPlugin(SupervisedPlugin, supports_distributed=True):
    """
    Experience replay plugin.

    Handles an external memory filled with randomly selected
    patterns and implementing `before_training_exp` and `after_training_exp`
    callbacks.
    The `before_training_exp` callback is implemented in order to use the
    dataloader that creates mini-batches with examples from both training
    data and external memory. The examples in the mini-batch is balanced
    such that there are the same number of examples for each experience.

    The `after_training_exp` callback is implemented in order to add new
    patterns to the external memory.

    The :mem_size: attribute controls the total number of patterns to be stored
    in the external memory.

    :param batch_size: the size of the data batch. If set to `None`, it
        will be set equal to the strategy's batch size.
    :param batch_size_mem: the size of the memory batch. If
        `task_balanced_dataloader` is set to True, it must be greater than or
        equal to the number of tasks. If its value is set to `None`
        (the default value), it will be automatically set equal to the
        data batch size.
    :param task_balanced_dataloader: if True, buffer data loaders will be
            task-balanced, otherwise it will create a single dataloader for the
            buffer samples.
    :param storage_policy: The policy that controls how to add new exemplars
                           in memory
    """

    def __init__(
        self,
        mem_size: int = 200,
        batch_size: Optional[int] = None,
        batch_size_mem: Optional[int] = None,
        task_balanced_dataloader: bool = False,
        storage_policy: Optional["ExemplarsBuffer"] = None,
    ):
        super().__init__()
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader

        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy = ExperienceBalancedBuffer(
                max_size=self.mem_size, adaptive_size=True
            )

    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        **kwargs
    ):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """
        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        # print("[DEBUG] Param hash BEFORE training exp 0:", get_model_param_hash(strategy.model))
        # print(f"[DEBUG] RNG state (first 5): {torch.get_rng_state()[:5]}")
        # dataset_ids = [i for i in range(len(strategy.experience.dataset))]
        # print("[DEBUG] Dataset sample indices (exp 0):", dataset_ids[:10])            
        # print("[DEBUG] Optimizer hash at start:", get_optimizer_hash(strategy.optimizer))
        
        batch_size = self.batch_size
        if batch_size is None:
            batch_size = strategy.train_mb_size

        batch_size_mem = self.batch_size_mem
        if batch_size_mem is None:
            batch_size_mem = strategy.train_mb_size

        assert strategy.adapted_dataset is not None

        other_dataloader_args = dict()

        if "ffcv_args" in kwargs:
            other_dataloader_args["ffcv_args"] = kwargs["ffcv_args"]

        if "persistent_workers" in kwargs:
            if parse(torch.__version__) >= parse("1.7.0"):
                other_dataloader_args["persistent_workers"] = kwargs[
                    "persistent_workers"
                ]
        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.storage_policy.buffer,
            oversample_small_tasks=False,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            task_balanced_dataloader=self.task_balanced_dataloader,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            **other_dataloader_args
        )
    # def before_training_iteration(self, strategy, **kwargs):
    #     if len(self.storage_policy.buffer) == 0:
    #         print(f"[DEBUG] Batch X hash at iter {strategy.clock.train_iterations}: {batch_hash(strategy.mb_x)}")
    #         print("[DEBUG] Is buffer empty during exp 0? ", len(self.storage_policy.buffer) == 0)
    #         return
        
    # def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
    #     if len(self.storage_policy.buffer) == 0:
    #         print("[DEBUG] Param hash AFTER training exp 0:", get_model_param_hash(strategy.model))
    #         print("[DEBUG] Param hash state dict AFTER training exp 0 and clustering:", hash_model_params(strategy.model))

    #     self.storage_policy.update(strategy, **kwargs)
        
    # def before_eval(self, strategy, **kwargs):
    #     print("[DEBUG] Model eval mode:", strategy.model.training == False)
            
    # def after_eval(self, strategy, **kwargs):
    #     print("[DEBUG] Inside after_eval hook")
    #     print("[DEBUG] Param hash AFTER eval exp :", get_model_param_hash(strategy.model))
    #     print("[DEBUG] Param hash state dict AFTER training exp and clustering:", hash_model_params(strategy.model))        
    #     # Print out first few predictions and targets from the last batch
    #     if hasattr(strategy, 'mb_output') and hasattr(strategy, 'mb_y'):
    #         preds = strategy.mb_output.detach().cpu()
    #         targets = strategy.mb_y.detach().cpu()
    #         print(f"[DEBUG] Eval preds (first 5): {preds[:5]}")
    #         print(f"[DEBUG] Eval targets (first 5): {targets[:5]}")
    #     else:
    #         print("[DEBUG] Eval outputs not available in strategy.")