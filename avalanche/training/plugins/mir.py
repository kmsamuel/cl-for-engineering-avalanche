import copy
from typing import TYPE_CHECKING
import torch
from avalanche.benchmarks.utils import concat_datasets
from avalanche.models.utils import avalanche_forward_base
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


# def update_temp(model, grad, lr):
#     model_copy = copy.deepcopy(model)
#     for g, p in zip(grad, model_copy.parameters()):
#         if g is not None:
#             p.data = p.data - lr * g
#     return model_copy

def update_temp(model, grad, lr):
    model_copy = copy.deepcopy(model)
    
    # Get only parameters that require gradients
    trainable_params = [p for p in model_copy.parameters() if p.requires_grad]
    
    # Update only if we have the correct number of gradients
    if len(trainable_params) == len(grad):
        for g, p in zip(grad, trainable_params):
            if g is not None:  # Some gradients might be None
                p.data = p.data - lr * g
    
    return model_copy

class MIRPlugin(SupervisedPlugin):
    """
    Maximally Interfered Retrieval plugin,
    Implements the strategy defined in
    "Online Continual Learning with Maximally Interfered Retrieval"
    https://arxiv.org/abs/1908.04742

    This strategy has been designed and tested in the
    Online Setting (OnlineCLScenario). However, it
    can also be used in non-online scenarios
    """

    def __init__(
        self,
        batch_size_mem: int,
        mem_size: int = 200,
        subsample: int = 200,
    ):
        """
        mem_size: int       : Fixed memory size
        subsample: int      : Size of the sample from which to look
                              for highest interfering exemplars
        batch_size_mem: int : Size of the batch sampled from
                              the bigger subsample batch
        """
        super().__init__()
        self.mem_size = mem_size
        self.subsample = subsample
        self.batch_size_mem = batch_size_mem
        self.storage_policy = ClassBalancedBuffer(
            max_size=self.mem_size, adaptive_size=True
        )
        self.replay_loader = None

    def before_backward(self, strategy, **kwargs):
        if self.replay_loader is None:
            return
        samples_x, samples_y, samples_tid, _ = next(self.replay_loader)
        print('test replay loader')
        print(samples_x)
        print(samples_y)
        print(samples_tid)
        # print(temp)
        
        # samples_x, samples_y, samples_tid = next(self.replay_loader)
        samples_x, samples_y, samples_tid = (
            samples_x.to(strategy.device),
            samples_y.to(strategy.device),
            samples_tid.to(strategy.device),
        )
        # Perform the temporary update with current data
        grad = torch.autograd.grad(
            strategy.loss,
            strategy.model.parameters(),
            retain_graph=True,
            allow_unused=True,
        )
        model_updated = update_temp(
            strategy.model, grad, strategy.optimizer.param_groups[0]["lr"]
        )
        # Selection of the most interfering samples, no grad required
        # plus we put the model in eval mode so that the additional
        # forward pass don't influence the batch norm statistics
        # strategy.model.eval()
        # model_updated.eval()
        with torch.no_grad():
            _old_red_strategy = strategy._criterion.reduction
            strategy._criterion.reduction = "none"
            old_output = avalanche_forward(strategy.model, samples_x, samples_tid)
            old_loss = strategy._criterion(old_output, samples_y)
            new_output = avalanche_forward(model_updated, samples_x, samples_tid)
            new_loss = strategy._criterion(new_output, samples_y)
            loss_diff = new_loss - old_loss
            chosen_samples_indexes = torch.argsort(loss_diff)[
                len(samples_x) - self.batch_size_mem :
            ]
            strategy._criterion.reduction = _old_red_strategy
        # strategy.model.train()
        # Choose the samples and add their loss to the current loss
        chosen_samples_x, chosen_samples_y, chosen_samples_tid = (
            samples_x[chosen_samples_indexes],
            samples_y[chosen_samples_indexes],
            samples_tid[chosen_samples_indexes],
        )
        replay_output = avalanche_forward(
            strategy.model, chosen_samples_x, chosen_samples_tid
        )
        replay_loss = strategy._criterion(replay_output, chosen_samples_y)
        strategy.loss += replay_loss

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        self.storage_policy.update(strategy, **kwargs)
        # Exclude classes that were in the last batch
        buffer = concat_datasets(
            [
                self.storage_policy.buffer_groups[key].buffer
                for key, _ in self.storage_policy.buffer_groups.items()
                if int(key) not in torch.unique(strategy.mb_y).cpu()
            ]
        )
        if len(buffer) > self.batch_size_mem:
            self.replay_loader = cycle(
                torch.utils.data.DataLoader(
                    buffer,
                    batch_size=self.subsample,
                    shuffle=True,
                )
            )
        else:
            self.replay_loader = None


class RegressionMIRPlugin(SupervisedPlugin):
    """
    Maximally Interfered Retrieval plugin adapted for regression tasks,
    Based on "Online Continual Learning with Maximally Interfered Retrieval"
    https://arxiv.org/abs/1908.04742

    This adaptation handles regression tasks with binned targets.
    """

    def __init__(
        self,
        batch_size_mem: int,
        mem_size: int = 200,
        subsample: int = 200,
    ):
        """
        mem_size: int       : Fixed memory size
        subsample: int      : Size of the sample from which to look
                              for highest interfering exemplars
        batch_size_mem: int : Size of the batch sampled from
                              the bigger subsample batch
        """
        super().__init__()
        self.mem_size = mem_size
        self.subsample = subsample
        self.batch_size_mem = batch_size_mem
        self.storage_policy = ClassBalancedBuffer(
            max_size=self.mem_size, adaptive_size=True
        )
        self.replay_loader = None

    def before_backward(self, strategy, **kwargs):
        if self.replay_loader is None:
            return
            
        # Get next batch from replay loader
        replay_data = next(self.replay_loader)
        
        # Handle both 3-value and 4-value returns from data loader
        if len(replay_data) == 3:
            samples_x, samples_y, samples_tid = replay_data
        elif len(replay_data) == 4:
            # If the loader returns 4 values (like in the DER implementation with logits)
            samples_x, samples_y, samples_tid, _ = replay_data
        else:
            raise ValueError(f"Unexpected number of values from replay_loader: {len(replay_data)}")
            
        samples_x, samples_y, samples_tid = (
            samples_x.to(strategy.device),
            samples_y.to(strategy.device),
            samples_tid.to(strategy.device),
        )
        
        # Check if loss requires grad
        if not strategy.loss.requires_grad:
            # If loss doesn't require grad, we can't compute gradients
            # This can happen if there are non-differentiable operations in the model
            # or if the model parameters don't have requires_grad=True
            # In this case, we'll fall back to a simpler approach: just use random samples
            
            # If subsample is larger than batch_size_mem, select random indices
            if len(samples_x) > self.batch_size_mem:
                # Select random indices
                indices = torch.randperm(len(samples_x))[:self.batch_size_mem]
                chosen_samples_x = samples_x[indices]
                chosen_samples_y = samples_y[indices]
                chosen_samples_tid = samples_tid[indices]
            else:
                # Use all samples if we have fewer than batch_size_mem
                chosen_samples_x = samples_x
                chosen_samples_y = samples_y
                chosen_samples_tid = samples_tid
                
            # Compute replay loss with these samples
            replay_output = avalanche_forward_base(
                strategy.model, chosen_samples_x, chosen_samples_tid
            )
            replay_loss = strategy._criterion(replay_output, chosen_samples_y)
            strategy.loss += replay_loss
            return
        
        # try:
        #     # Try the original MIR approach with gradient computation
        #     # Perform the temporary update with current data
        grad = torch.autograd.grad(
            strategy.loss,
            [p for p in strategy.model.parameters() if p.requires_grad],
            retain_graph=True,
            allow_unused=True,
        )
        
        # Filter out None gradients
        valid_grad = [g for g in grad if g is not None]
        valid_params = [p for p in strategy.model.parameters() if p.requires_grad]
        
        model_updated = update_temp(
            strategy.model, valid_grad, strategy.optimizer.param_groups[0]["lr"]
        )
        
        # Selection of the most interfering samples, no grad required
        with torch.no_grad():
            _old_red_strategy = strategy._criterion.reduction
            strategy._criterion.reduction = "none"
            old_output = avalanche_forward_base(strategy.model, samples_x, samples_tid)
            old_loss = strategy._criterion(old_output, samples_y)
            new_output = avalanche_forward_base(model_updated, samples_x, samples_tid)
            new_loss = strategy._criterion(new_output, samples_y)
            loss_diff = new_loss - old_loss
            
            if len(samples_x) <= self.batch_size_mem:
                # Use all samples if we have fewer than batch_size_mem
                chosen_samples_indexes = torch.arange(len(samples_x))
            else:
                # Select the samples with the highest loss difference
                chosen_samples_indexes = torch.argsort(loss_diff)[
                    len(samples_x) - self.batch_size_mem :
                ]
            
            strategy._criterion.reduction = _old_red_strategy
            
        # Choose the samples and add their loss to the current loss

        
        chosen_samples_x, chosen_samples_y, chosen_samples_tid = (
            samples_x[chosen_samples_indexes],
            samples_y[chosen_samples_indexes],
            samples_tid[chosen_samples_indexes],
        )
        # print(f"samples_x shape: {samples_x.shape}")
        # print(chosen_samples_x)

        # print(f"chosen_samples_x shape: {chosen_samples_x.shape}")
        # print(chosen_samples_x)
        # chosen_samples_x = chosen_samples_x.squeeze(1)  # This should make it [32, 4, 20000]
        # print(f"chosen_samples_x shape after reshape: {chosen_samples_x.shape}")
        # print(chosen_samples_x)

                    
        # except RuntimeError as e:
        #     If any error occurs during gradient computation, fall back to random samples
        # print(f"Warning: MIR gradient computation failed, using random selection instead. Error: {e}")
        
        # # If subsample is larger than batch_size_mem, select random indices
        if len(samples_x) > self.batch_size_mem:
            # Select random indices
            indices = torch.randperm(len(samples_x))[:self.batch_size_mem]
            chosen_samples_x = samples_x[indices]
            chosen_samples_y = samples_y[indices]
            chosen_samples_tid = samples_tid[indices]
        else:
            # Use all samples if we have fewer than batch_size_mem
            chosen_samples_x = samples_x
            chosen_samples_y = samples_y
            chosen_samples_tid = samples_tid
        
        # Compute replay loss with the selected samples
        replay_output = avalanche_forward_base(
            strategy.model, chosen_samples_x, chosen_samples_tid
        )
        replay_loss = strategy._criterion(replay_output, chosen_samples_y)
        strategy.loss += replay_loss

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        self.storage_policy.update(strategy, **kwargs)
        
        # Exclude bins that were in the last batch
        storage_groups = self.storage_policy.buffer_groups
        
        # Get unique bins from the current minibatch
        current_bins = torch.unique(strategy.mb_y).cpu()
        
        # Create a buffer excluding bins from the current minibatch
        buffer_datasets = [
            storage_groups[key].buffer
            for key in storage_groups.keys()
            if int(key) not in current_bins
        ]
        
        if buffer_datasets:
            buffer = concat_datasets(buffer_datasets)
            if len(buffer) > self.batch_size_mem:
                self.replay_loader = cycle(
                    torch.utils.data.DataLoader(
                        buffer,
                        batch_size=self.subsample,
                        shuffle=True,
                    )
                )
                return
        
        self.replay_loader = None
        
__all__ = ["MIRPlugin", "RegressionMIRPlugin"]
