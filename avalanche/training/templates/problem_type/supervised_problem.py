from avalanche.models.utils import avalanche_forward, avalanche_forward_base
from avalanche.training.templates.strategy_mixin_protocol import (
    SupervisedStrategyProtocol,
    TSGDExperienceType,
    TMBInput,
    TMBOutput,
)


# Types are perfectly ok for MyPy
# Also confirmed here: https://stackoverflow.com/a/70907644
# PyLance just does not understand it
class SupervisedProblem(
    SupervisedStrategyProtocol[TSGDExperienceType, TMBInput, TMBOutput]
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def mb_x(self):
        """Current mini-batch input voxels."""
        mbatch = self.mbatch
        assert mbatch is not None
        return mbatch[0]

    @property
    def mb_f(self):
        """Current mini-batch flight conditions."""
        mbatch = self.mbatch
        assert mbatch is not None
        return mbatch[1]

    @property
    def mb_y(self):
        """Current mini-batch target."""
        mbatch = self.mbatch
        assert mbatch is not None
        return mbatch[1]

    @property
    def mb_task_id(self):
        """Current mini-batch task labels."""
        mbatch = self.mbatch
        assert mbatch is not None
        assert len(mbatch) >= 3
        return mbatch[-1]

    def criterion(self):
        """Loss function for supervised problems."""
        
        return self._criterion(self.mb_output, self.mb_y)

    def forward_base(self):
        """Compute the model's output given the current mini-batch."""
        return avalanche_forward_base(self.model, self.mb_x, self.mb_task_id)

    def forward(self):
        """Compute the model's output given the current mini-batch."""
        return avalanche_forward(self.model, self.mb_x, self.mb_f, self.mb_task_id)



    def _unpack_minibatch(self):
        """Check if the current mini-batch has 3 components."""
        mbatch = self.mbatch
        assert mbatch is not None
        assert len(mbatch) >= 3

        if isinstance(mbatch, tuple):
            mbatch = list(mbatch)
            self.mbatch = mbatch

        def _move_to_device(obj, device, non_blocking=True):
        # Added Feb 2025 to handle dictionary input data (point clouds + flight conditions)
            """Move tensors or dictionaries of tensors to the specified device."""
            if hasattr(obj, 'to'):  # Check if object has a 'to' method (like tensors)
                return obj.to(device, non_blocking=non_blocking)
            elif isinstance(obj, dict):
                return {k: _move_to_device(v, device, non_blocking) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_move_to_device(x, device, non_blocking) for x in obj]
            elif isinstance(obj, tuple):
                return tuple(_move_to_device(x, device, non_blocking) for x in obj)
            else:
                return obj

        for i in range(len(mbatch)):
            # print("Minibatch device before transfer:", getattr(mbatch[0], 'device', 'no device'))
            # mbatch[i] = mbatch[i].to(self.device, non_blocking=True)  # type: ignore
            if getattr(mbatch[i], 'device', 'cpu') != self.device:
                mbatch[i] = _move_to_device(mbatch[i], self.device, non_blocking=True)
               # mbatch[i] = mbatch[i].to(self.device, non_blocking=True)


__all__ = ["SupervisedProblem"]
