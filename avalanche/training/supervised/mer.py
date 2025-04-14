from copy import deepcopy
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Module, MSELoss
from torch.optim import Optimizer

from avalanche.models.utils import avalanche_forward, avalanche_forward_base
from avalanche.training.plugins import EvaluationPlugin, SupervisedPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.training.templates import SupervisedMetaLearningTemplate
from avalanche.training.templates.strategy_mixin_protocol import CriterionType


class MERBuffer:
    def __init__(self, mem_size=100, batch_size_mem=10, device=torch.device("cpu")):
        self.storage_policy = ReservoirSamplingBuffer(max_size=mem_size)
        self.batch_size_mem = batch_size_mem
        self.device = device

    def update(self, strategy):
        self.storage_policy.update(strategy)

    def __len__(self):
        return len(self.storage_policy.buffer)

    def get_batch(self, x, y, t):
        if len(self) == 0:
            return x, y, t

        bsize = min(len(self), self.batch_size_mem)
        rnd_ind = torch.randperm(len(self))[:bsize].tolist()
        buff_x = torch.cat(
            [self.storage_policy.buffer[i][0].unsqueeze(0) for i in rnd_ind]
        ).to(self.device)
        buff_y = torch.LongTensor(
            [self.storage_policy.buffer[i][1] for i in rnd_ind]
        ).to(self.device)
        buff_t = torch.LongTensor(
            [self.storage_policy.buffer[i][2] for i in rnd_ind]
        ).to(self.device)

        mixed_x = torch.cat([x, buff_x], dim=0)
        mixed_y = torch.cat([y, buff_y], dim=0)
        mixed_t = torch.cat([t, buff_t], dim=0)

        return mixed_x, mixed_y, mixed_t


class MER(SupervisedMetaLearningTemplate):
    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        # criterion: CriterionType = CrossEntropyLoss(),
        criterion: CriterionType = MSELoss(),
        scheduler_type=None,
        mem_size=200,
        batch_size_mem=10,
        n_inner_steps=5,
        beta=0.1,
        gamma=0.1,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = 1,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[Sequence["SupervisedPlugin"]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        peval_mode="epoch",
        **kwargs
    ):
        """Implementation of Look-ahead MAML (LaMAML) algorithm in Avalanche
            using Higher library for applying fast updates.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param mem_size: maximum size of the buffer.
        :param batch_size_mem: number of samples to retrieve from buffer
            for each sample.
        :param n_inner_steps: number of inner updates per sample.
        :param beta: coefficient for within-batch Reptile update.
        :param gamma: coefficient for within-task Reptile update.

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

        self.buffer = MERBuffer(
            mem_size=mem_size,
            batch_size_mem=batch_size_mem,
            device=self.device,
        )
        self.n_inner_steps = n_inner_steps
        self.beta = beta
        self.gamma = gamma

        # --- START OF SCHEDULER ADDITION ---
        self.scheduler_type = scheduler_type
        self.current_epoch = 0
        
        # We only support manual scheduler for this implementation
        if self.scheduler_type == 'manual':
            self.scheduler = None
        else:
            self.scheduler = None
        # --- END OF SCHEDULER ADDITION ---

    # --- START OF SCHEDULER ADDITION ---
    def adjust_learning_rate(self, optimizer, epoch):
        initial_lr = 0.001
        
        # epoch_in_cycle = epoch % 200  # 200 for drivaernet point clouds
        epoch_in_cycle = epoch % 150  # 200 for drivaernet point clouds
        
        # if epoch_in_cycle >= 150:  # 150 for drivaernet PC
        if epoch_in_cycle >= 100:  # 100 for drivaernet Par
        #     current_lr = initial_lr * (0.1 ** 3)
        # elif epoch_in_cycle >= 100:
            current_lr = initial_lr * (0.1 ** 2)
        elif epoch_in_cycle >= 50:
            current_lr = initial_lr * 0.1
        else:
            current_lr = initial_lr        
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
    # --- END OF SCHEDULER ADDITION ---
    
    def forward(self):
        # Call a specific forward implementation
        return self.specific_forward_method()

    def specific_forward_method(self):
        # Your custom forward logic
        return avalanche_forward_base(self.model, self.mb_x, self.mb_task_id)
    
    def _before_inner_updates(self, **kwargs):
        self.w_bef = deepcopy(self.model.state_dict())
        super()._before_inner_updates(**kwargs)

    # def _inner_updates(self, **kwargs):
    #     for inner_itr in range(self.n_inner_steps):
    #         x, y, t = self.mb_x, self.mb_y, self.mb_task_id
    #         x, y, t = self.buffer.get_batch(x, y, t)

    #         # Inner updates
    #         w_bef_t = deepcopy(self.model.state_dict())
    #         for idx in range(x.shape[0]):
    #             x_b = x[idx].unsqueeze(0)
    #             y_b = y[idx].unsqueeze(0)
    #             t_b = t[idx].unsqueeze(0)
    #             self.model.zero_grad()
    #             pred = avalanche_forward_base(self.model, x_b, t_b)
    #             loss = self._criterion(pred, y_b)
    #             loss.backward()
    #             self.optimizer.step()

    #         # Within-batch Reptile update
    #         w_aft_t = self.model.state_dict()
    #         load_dict = {}
    #         for name, param in self.model.named_parameters():
    #             load_dict[name] = w_bef_t[name] + (
    #                 (w_aft_t[name] - w_bef_t[name]) * self.beta
    #             )

    #         self.model.load_state_dict(load_dict, strict=False)

    def _inner_updates(self, **kwargs): ## account for batch normalization issue --> do the method with model.eval
        # Store original training mode
        training_mode = self.model.training
        
        for inner_itr in range(self.n_inner_steps):
            x, y, t = self.mb_x, self.mb_y, self.mb_task_id
            x, y, t = self.buffer.get_batch(x, y, t)

            # Inner updates
            w_bef_t = deepcopy(self.model.state_dict())
            for idx in range(x.shape[0]):
                # Set BatchNorm to eval mode but keep everything else in train mode
                self.model.eval()  # This sets batch norm to use running statistics
                
                x_b = x[idx].unsqueeze(0)
                y_b = y[idx].unsqueeze(0)
                t_b = t[idx].unsqueeze(0)
                self.model.zero_grad()
                pred = avalanche_forward_base(self.model, x_b, t_b)
                loss = self._criterion(pred, y_b)
                loss.backward()
                self.optimizer.step()

            # Within-batch Reptile update
            w_aft_t = self.model.state_dict()
            load_dict = {}
            for name, param in self.model.named_parameters():
                load_dict[name] = w_bef_t[name] + (
                    (w_aft_t[name] - w_bef_t[name]) * self.beta
                )

            self.model.load_state_dict(load_dict, strict=False)
        
        # Restore original training mode
        if training_mode:
            self.model.train()
            
    def _outer_update(self, **kwargs):
        w_aft = self.model.state_dict()

        load_dict = {}
        for name, param in self.model.named_parameters():
            load_dict[name] = self.w_bef[name] + (
                (w_aft[name] - self.w_bef[name]) * self.gamma
            )

        self.model.load_state_dict(load_dict, strict=False)

        with torch.no_grad():
            pred = self.forward()
            self.loss = self._criterion(pred, self.mb_y)

    def _after_training_exp(self, **kwargs):
        self.buffer.update(self)
        super()._after_training_exp(**kwargs)

    def training_epoch(self, **kwargs):
        """Training epoch with manual learning rate scheduling."""
        # Apply manual learning rate scheduling if enabled
        if self.scheduler_type == 'manual':
            self.adjust_learning_rate(self.optimizer, self.current_epoch)
            
        # Display current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f'**Current LR**: {current_lr:.6f}')
        
        # Call the parent class training_epoch method
        super().training_epoch(**kwargs)
        
        # Increment current epoch
        self.current_epoch += 1