from collections import defaultdict
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    SupportsInt,
    Union,
)

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer

from avalanche.benchmarks.utils import make_avalanche_dataset
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.data_attribute import TensorDataAttribute
from avalanche.benchmarks.utils.flat_data import FlatData
from avalanche.training.templates.strategy_mixin_protocol import CriterionType
from avalanche.training.utils import cycle
from avalanche.core import SupervisedPlugin
from avalanche.training.plugins.evaluation import (
    EvaluationPlugin,
    default_evaluator,
)
from avalanche.training.storage_policy import (
    BalancedExemplarsBuffer,
    ReservoirSamplingBuffer,
)
from avalanche.training.templates import SupervisedTemplate
from avalanche.models.utils import avalanche_forward, avalanche_forward_base


@torch.no_grad()
def compute_dataset_logits(dataset, model, batch_size, device, num_workers=0):
    was_training = model.training
    model.eval()

    logits = []
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    for x, _, _ in loader:
        x = x.to(device)
        out = model(x)
        out = out.detach().cpu()

        for row in out:
            logits.append(torch.clone(row))

    if was_training:
        model.train()

    return logits


class ClassBalancedBufferWithLogits(BalancedExemplarsBuffer):
    """
    ClassBalancedBuffer that also stores the logits
    """

    def __init__(
        self,
        max_size: int,
        adaptive_size: bool = True,
        total_num_classes: Optional[int] = None,
    ):
        """Init.

        :param max_size: The max capacity of the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all
                            observed experiences (keys in replay_mem).
        :param total_num_classes: If adaptive size is False, the fixed number
                                  of classes to divide capacity over.
        :param transforms: transformation to be applied to the buffer
        """
        if not adaptive_size:
            assert (
                total_num_classes is not None and total_num_classes > 0
            ), "When fixed exp mem size, total_num_classes should be > 0."

        super().__init__(max_size, adaptive_size, total_num_classes)
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.seen_classes: Set[int] = set()

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        assert strategy.experience is not None
        new_data: AvalancheDataset = strategy.experience.dataset

        logits = compute_dataset_logits(
            new_data.eval(),
            strategy.model,
            strategy.train_mb_size,
            strategy.device,
            num_workers=kwargs.get("num_workers", 0),
        )
        new_data_with_logits = make_avalanche_dataset(
            new_data,
            data_attributes=[
                TensorDataAttribute(
                    FlatData([logits], discard_elements_not_in_indices=True),
                    name="logits",
                    use_in_getitem=True,
                )
            ],
        )
        # Get sample idxs per class
        cl_idxs: Dict[int, List[int]] = defaultdict(list)
        targets: Sequence[SupportsInt] = getattr(new_data, "targets")
        for idx, target in enumerate(targets):
            # Conversion to int may fix issues when target
            # is a single-element torch.tensor
            target = int(target)
            cl_idxs[target].append(idx)

        # Make AvalancheSubset per class
        cl_datasets = {}
        for c, c_idxs in cl_idxs.items():
            subset = new_data_with_logits.subset(c_idxs)
            cl_datasets[c] = subset
        # Update seen classes
        self.seen_classes.update(cl_datasets.keys())

        # associate lengths to classes
        lens = self.get_group_lengths(len(self.seen_classes))
        class_to_len = {}
        for class_id, ll in zip(self.seen_classes, lens):
            class_to_len[class_id] = ll

        # update buffers with new data
        for class_id, new_data_c in cl_datasets.items():
            ll = class_to_len[class_id]
            if class_id in self.buffer_groups:
                old_buffer_c = self.buffer_groups[class_id]
                # Here it uses underlying dataset
                old_buffer_c.update_from_dataset(new_data_c)
                old_buffer_c.resize(strategy, ll)
            else:
                new_buffer = ReservoirSamplingBuffer(ll)
                new_buffer.update_from_dataset(new_data_c)
                self.buffer_groups[class_id] = new_buffer

        # resize buffers
        for class_id, class_buf in self.buffer_groups.items():
            self.buffer_groups[class_id].resize(strategy, class_to_len[class_id])


class DER(SupervisedTemplate):
    """
    Implements the DER and the DER++ Strategy,
    from the "Dark Experience For General Continual Learning"
    paper, Buzzega et. al, https://arxiv.org/abs/2004.07211
    """

    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType = CrossEntropyLoss(),
        mem_size: int = 200,
        batch_size_mem: Optional[int] = None,
        alpha: float = 0.1,
        beta: float = 0.5,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        peval_mode="epoch",
        **kwargs
    ):
        """
        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param mem_size: int       : Fixed memory size
        :param batch_size_mem: int : Size of the batch sampled from the buffer
        :param alpha: float : Hyperparameter weighting the MSE loss
        :param beta: float : Hyperparameter weighting the CE loss,
                             when more than 0, DER++ is used instead of DER
        :param transforms: Callable: Transformations to use for
                                     both the dataset and the buffer data, on
                                     top of already existing
                                     test transformations.
                                     If any supplementary transformations
                                     are applied to the
                                     input data, it will be
                                     overwritten by this argument
        :param train_mb_size: mini-batch size for training.
        :param train_passes: number of training passes.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` experiences and at the end of
            the learning experience.
        :param peval_mode: one of {'experience', 'iteration'}. Decides whether
            the periodic evaluation during training should execute every
            `eval_every` experience or iterations (Default='experience').
        """
        super().__init__(
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
            peval_mode=peval_mode,
            **kwargs
        )
        if batch_size_mem is None:
            self.batch_size_mem = train_mb_size
        else:
            self.batch_size_mem = batch_size_mem
        self.mem_size = mem_size
        self.storage_policy = ClassBalancedBufferWithLogits(
            self.mem_size, adaptive_size=True
        )
        self.replay_loader = None
        self.alpha = alpha
        self.beta = beta

    def _before_training_exp(self, **kwargs):
        buffer = self.storage_policy.buffer
        if len(buffer) >= self.batch_size_mem:
            self.replay_loader = cycle(
                torch.utils.data.DataLoader(
                    buffer,
                    batch_size=self.batch_size_mem,
                    shuffle=True,
                    drop_last=True,
                    num_workers=kwargs.get("num_workers", 0),
                )
            )
        else:
            self.replay_loader = None

        super()._before_training_exp(**kwargs)

    def _after_training_exp(self, **kwargs):
        self.replay_loader = None  # Allow DER to be checkpointed
        self.storage_policy.update(self, **kwargs)
        super()._after_training_exp(**kwargs)

    def _before_forward(self, **kwargs):
        super()._before_forward(**kwargs)
        if self.replay_loader is None:
            return None

        batch_x, batch_y, batch_tid, batch_logits = next(self.replay_loader)
        batch_x, batch_y, batch_tid, batch_logits = (
            batch_x.to(self.device),
            batch_y.to(self.device),
            batch_tid.to(self.device),
            batch_logits.to(self.device),
        )
        self.mbatch[0] = torch.cat((batch_x, self.mbatch[0]))
        self.mbatch[1] = torch.cat((batch_y, self.mbatch[1]))
        self.mbatch[2] = torch.cat((batch_tid, self.mbatch[2]))
        self.batch_logits = batch_logits

    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = self._make_empty_loss()

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            if self.replay_loader is not None:
                # DER Loss computation

                self.loss += F.cross_entropy(
                    self.mb_output[self.batch_size_mem :],
                    self.mb_y[self.batch_size_mem :],
                )

                self.loss += self.alpha * F.mse_loss(
                    self.mb_output[: self.batch_size_mem],
                    self.batch_logits,
                )
                self.loss += self.beta * F.cross_entropy(
                    self.mb_output[: self.batch_size_mem],
                    self.mb_y[: self.batch_size_mem],
                )

                # They are a few difference compared to the autors impl:
                # - Joint forward pass vs. 3 forward passes
                # - One replay batch vs two replay batches
                # - Logits are stored from the non-transformed sample
                #   after training on task vs instantly on transformed sample

            else:
                self.loss += self.criterion()

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

class RegressionBalancedBufferWithLogits(BalancedExemplarsBuffer):
    """
    A buffer for regression tasks that balances samples across different bins of regression values.
    This is a modified version of ClassBalancedBufferWithLogits that works with regression targets
    that have been binned into discrete categories.
    """

    def __init__(
        self,
        max_size: int,
        adaptive_size: bool = True,
        total_num_bins: Optional[int] = None,
    ):
        """Init.

        :param max_size: The max capacity of the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all
                            observed bins (keys in replay_mem).
        :param total_num_bins: If adaptive size is False, the fixed number
                              of bins to divide capacity over.
        """
        if not adaptive_size:
            assert (
                total_num_bins is not None and total_num_bins > 0
            ), "When fixed exp mem size, total_num_bins should be > 0."

        super().__init__(max_size, adaptive_size, total_num_bins)
        self.adaptive_size = adaptive_size
        self.total_num_bins = total_num_bins
        self.seen_bins: Set[int] = set()  # Renamed from seen_classes to seen_bins

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        assert strategy.experience is not None
        new_data: AvalancheDataset = strategy.experience.dataset

        # Compute and store logits for the new data
        logits = compute_dataset_logits(
            new_data.eval(),
            strategy.model,
            strategy.train_mb_size,
            strategy.device,
            num_workers=kwargs.get("num_workers", 0),
        )
        new_data_with_logits = make_avalanche_dataset(
            new_data,
            data_attributes=[
                TensorDataAttribute(
                    FlatData([logits], discard_elements_not_in_indices=True),
                    name="logits",
                    use_in_getitem=True,
                )
            ],
        )
        
        # Get sample indices per bin
        bin_idxs: Dict[int, List[int]] = defaultdict(list)
        targets: Sequence[SupportsInt] = getattr(new_data, "targets")
        for idx, target in enumerate(targets):
            # Conversion to int is critical for binned regression values
            bin_id = int(target)
            bin_idxs[bin_id].append(idx)

        # Make AvalancheSubset per bin
        bin_datasets = {}
        for bin_id, bin_idxs in bin_idxs.items():
            subset = new_data_with_logits.subset(bin_idxs)
            bin_datasets[bin_id] = subset
            
        # Update seen bins
        self.seen_bins.update(bin_datasets.keys())

        # Associate lengths to bins
        lens = self.get_group_lengths(len(self.seen_bins))
        bin_to_len = {}
        for bin_id, ll in zip(self.seen_bins, lens):
            bin_to_len[bin_id] = ll

        # Update buffers with new data
        for bin_id, new_data_bin in bin_datasets.items():
            ll = bin_to_len[bin_id]
            if bin_id in self.buffer_groups:
                old_buffer_bin = self.buffer_groups[bin_id]
                old_buffer_bin.update_from_dataset(new_data_bin)
                old_buffer_bin.resize(strategy, ll)
            else:
                new_buffer = ReservoirSamplingBuffer(ll)
                new_buffer.update_from_dataset(new_data_bin)
                self.buffer_groups[bin_id] = new_buffer

        # Resize buffers
        for bin_id, bin_buf in self.buffer_groups.items():
            self.buffer_groups[bin_id].resize(strategy, bin_to_len[bin_id])


# Modified DER implementation that uses the RegressionBalancedBufferWithLogits
class RegressionDER(SupervisedTemplate):
    """
    Implements the DER strategy adapted for regression tasks,
    based on "Dark Experience For General Continual Learning"
    paper, Buzzega et. al, https://arxiv.org/abs/2004.07211
    
    This version works with regression targets that have been binned into discrete categories.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Union[torch.nn.Module, callable] = torch.nn.MSELoss(),  # Changed default to MSE for regression
        scheduler_type=None,        
        mem_size: int = 200,
        batch_size_mem: Optional[int] = None,
        alpha: float = 0.1,
        beta: float = 0.5,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator=None,
        eval_every=-1,
        peval_mode="epoch",
        **kwargs
    ):
        """
        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function (defaults to MSELoss for regression).
        :param mem_size: int       : Fixed memory size
        :param batch_size_mem: int : Size of the batch sampled from the buffer
        :param alpha: float : Hyperparameter weighting the MSE loss between predictions and stored logits
        :param beta: float : Hyperparameter weighting the MSE loss between predictions and targets for memory samples
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging and metric computations.
        :param eval_every: frequency of the calls to `eval` inside the training loop.
        :param peval_mode: one of {'experience', 'iteration'}.
        """
        super().__init__(
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
            peval_mode=peval_mode,
            **kwargs
        )
        if batch_size_mem is None:
            self.batch_size_mem = train_mb_size
        else:
            self.batch_size_mem = batch_size_mem
        self.mem_size = mem_size
        
        # Use our regression-adapted buffer
        self.storage_policy = RegressionBalancedBufferWithLogits(
            self.mem_size, adaptive_size=True
        )
        self.replay_loader = None
        self.alpha = alpha
        self.beta = beta
        
        self.scheduler_type = scheduler_type
        self.current_epoch = 0
        
        # Set up appropriate scheduler based on type
        if self.scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            self.scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=5,  # Shorter first cycle to allow multiple restarts
                T_mult=2,  # Double the cycle length after each restart
                eta_min=5e-6  # Slightly higher minimum learning rate
            )
        else:  # Use manual adjustment
            self.scheduler = None

    def _before_training_exp(self, **kwargs):
        buffer = self.storage_policy.buffer
        if len(buffer) >= self.batch_size_mem:
            self.replay_loader = cycle(
                torch.utils.data.DataLoader(
                    buffer,
                    batch_size=self.batch_size_mem,
                    shuffle=True,
                    drop_last=True,
                    num_workers=kwargs.get("num_workers", 0),
                )
            )
        else:
            self.replay_loader = None

        super()._before_training_exp(**kwargs)

    def _after_training_exp(self, **kwargs):
        self.replay_loader = None  # Allow DER to be checkpointed
        self.storage_policy.update(self, **kwargs)
        super()._after_training_exp(**kwargs)

    def _before_forward(self, **kwargs):
        super()._before_forward(**kwargs)
        if self.replay_loader is None:
            return None

        batch_x, batch_y, batch_tid, batch_logits = next(self.replay_loader)
        batch_x, batch_y, batch_tid, batch_logits = (
            batch_x.to(self.device),
            batch_y.to(self.device),
            batch_tid.to(self.device),
            batch_logits.to(self.device),
        )
        self.mbatch[0] = torch.cat((batch_x, self.mbatch[0]))
        self.mbatch[1] = torch.cat((batch_y, self.mbatch[1]))
        self.mbatch[2] = torch.cat((batch_tid, self.mbatch[2]))
        self.batch_logits = batch_logits
        
    def adjust_learning_rate(self, optimizer, epoch):
        initial_lr = 0.001
        
        epoch_in_cycle = epoch % 200  # 200 for drivaernet point clouds
        # epoch_in_cycle = epoch % 150  # 200 for drivaernet point clouds
        
        if epoch_in_cycle >= 150:  # 150 for drivaernet PC
            current_lr = initial_lr * (0.1 ** 3)
        # if epoch_in_cycle >= 100:  # 100 for drivaernet Par
        elif epoch_in_cycle >= 100:
            current_lr = initial_lr * (0.1 ** 2)
        elif epoch_in_cycle >= 50:
            current_lr = initial_lr * 0.1
        else:
            current_lr = initial_lr        
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

    def forward(self):
        # Call a specific forward implementation
        return self.specific_forward_method()

    def specific_forward_method(self):
        # Your custom forward logic
        return avalanche_forward_base(self.model, self.mb_x, self.mb_task_id)

    def training_epoch(self, **kwargs):
        """Training epoch adapted for regression tasks with learning rate scheduling."""
        # Apply learning rate scheduling if needed
        if self.scheduler is None:
            # Use manual adjustment if no scheduler type specified
            if self.scheduler_type == 'manual':
                self.adjust_learning_rate(self.optimizer, self.current_epoch)
        else:
            # If using a PyTorch scheduler, step it here (for epoch-based schedulers)
            if hasattr(self.scheduler, 'step'):
                self.scheduler.step(self.current_epoch)
        
        # Display current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f'**Current LR**: {current_lr:.6f}')
        
        # Run the original training epoch
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = self._make_empty_loss()

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            if self.replay_loader is not None:
                # DER Loss computation for regression
                
                # Main loss on current batch (MSE for regression)
                self.loss += self._criterion(
                    self.mb_output[self.batch_size_mem :],
                    self.mb_y[self.batch_size_mem :],
                )

                # Dark knowledge distillation loss (keep predictions consistent with past)
                self.loss += self.alpha * torch.nn.functional.mse_loss(
                    self.mb_output[: self.batch_size_mem],
                    self.batch_logits,
                )
                
                # Optional rehearsal loss on memory samples
                self.loss += self.beta * self._criterion(
                    self.mb_output[: self.batch_size_mem],
                    self.mb_y[: self.batch_size_mem],
                )
            else:
                self.loss += self._criterion(self.mb_output, self.mb_y)

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)
            
        # Increment current epoch
        self.current_epoch += 1
            
__all__ = ["DER", "RegressionDER"]
