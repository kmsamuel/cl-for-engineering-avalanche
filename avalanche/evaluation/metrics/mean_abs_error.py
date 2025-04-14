################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 19-01-2021                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from typing import List, Optional, Union, Dict

import torch
from torch import Tensor
from avalanche.evaluation import Metric, GenericPluginMetric
from avalanche.evaluation.metrics.mean import Mean
from collections import defaultdict


class MeanAbsoluteError(Metric[float]):
    """Mean Absolute Error (MAE) metric. This is a standalone metric.

    The update method computes the MAE incrementally
    by keeping a running average of the <prediction, target> pairs
    of Tensors provided over time.

    Each time `result` is called, this metric emits the average MAE
    across all predictions made since the last `reset`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an MAE value of 0.
    """

    def __init__(self):
        """Creates an instance of the standalone MeanAbsoluteError metric.

        By default this metric in its initial state will return an MAE value
        of 0. The metric can be updated by using the `update` method
        while the running MAE can be retrieved using the `result` method.
        """
        self._mean_mae = Mean()
        """The mean utility that will be used to store the running MAE."""

    @torch.no_grad()
    def update(
        self,
        predicted_y: Tensor,
        true_y: Tensor,
    ) -> None:
        """Update the running MAE given the true and predicted labels.

        :param predicted_y: The model prediction.
        :param true_y: The ground truth.

        :return: None.
        """
        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")

        mae_value = torch.abs(predicted_y - true_y).mean().item()
        self._mean_mae.update(mae_value, len(true_y))

    def result(self) -> float:
        """Retrieves the running MAE.

        Calling this method will not change the internal state of the metric.

        :return: The current running MAE.
        """
        return self._mean_mae.result()

    def reset(self) -> None:
        """Resets the metric.

        :return: None.
        """
        self._mean_mae.reset()


class TaskAwareMeanAbsoluteError(Metric[Dict[int, float]]):
    """The task-aware Mean Absolute Error (MAE) metric.

    The metric computes a dictionary of <task_label, MAE value> pairs.
    update/result/reset methods are all task-aware.
    """

    def __init__(self):
        """Creates an instance of the task-aware MeanAbsoluteError metric."""
        self._mean_mae = defaultdict(MeanAbsoluteError)
        """
        The utility that will be used to store the running MAE
        for each task label.
        """

    @torch.no_grad()
    def update(
        self,
        predicted_y: Tensor,
        true_y: Tensor,
        task_labels: Union[int, Tensor],
    ) -> None:
        """Update the running MAE given the true and predicted labels.

        Parameter `task_labels` is used to decide how to update the inner
        dictionary: if int, only the dictionary value related to that task
        is updated. If Tensor, all the dictionary elements belonging to the
        task labels will be updated.

        :param predicted_y: The model prediction.
        :param true_y: The ground truth.
        :param task_labels: the task label associated with the current experience
            or a task labels vector.

        :return: None.
        """
        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")

        if isinstance(task_labels, Tensor) and len(task_labels) != len(true_y):
            raise ValueError("Size mismatch for true_y and task_labels tensors")

        if isinstance(task_labels, int):
            self._mean_mae[task_labels].update(predicted_y, true_y)
        elif isinstance(task_labels, Tensor):
            for pred, true, t in zip(predicted_y, true_y, task_labels):
                if isinstance(t, Tensor):
                    t = t.item()
                self._mean_mae[t].update(pred.unsqueeze(0), true.unsqueeze(0))
        else:
            raise ValueError(
                f"Task label type: {type(task_labels)}, expected int or Tensor"
            )

    def result(self, task_label: Optional[int] = None) -> Dict[int, float]:
        """
        Retrieves the running MAE.

        :param task_label: if None, return the entire dictionary of MAEs
            for each task. Otherwise return the dictionary
            `{task_label: MAE}`.
        :return: A dict of running MAEs for each task label.
        """
        assert task_label is None or isinstance(task_label, int)

        if task_label is None:
            return {k: v.result() for k, v in self._mean_mae.items()}
        else:
            return {task_label: self._mean_mae[task_label].result()}

    def reset(self, task_label: Optional[int] = None) -> None:
        """
        Resets the metric.

        :param task_label: if None, reset the entire dictionary.
            Otherwise, reset the value associated to `task_label`.

        :return: None.
        """
        assert task_label is None or isinstance(task_label, int)
        if task_label is None:
            self._mean_mae = defaultdict(MeanAbsoluteError)
        else:
            self._mean_mae[task_label].reset()
            
class ExperienceAwareMeanAbsoluteError(Metric[Union[float, Dict[int, float]]]):
    """
    Experience-Aware Mean Absolute Error (MAE) metric.
    
    This metric tracks MAE both globally and per-experience.
    When result() is called, it returns either the global MAE or
    the current experience's MAE based on the emit_at setting.
    """

    def __init__(self):
        """Creates an instance of the ExperienceAwareMeanAbsoluteError metric."""
        self._global_mae = Mean()
        self._exp_mae = defaultdict(Mean)
        self._current_exp = None
        self._emit_at = "stream"  # Default value, will be overridden

    @torch.no_grad()
    def update(
        self,
        predicted_y: Tensor,
        true_y: Tensor,
        experience_id: Optional[int] = None
    ) -> None:
        """
        Update the running MAE given the true and predicted labels.
        
        :param predicted_y: The model prediction.
        :param true_y: The ground truth.
        :param experience_id: The current experience ID.
        
        :return: None.
        """
        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")
            
        if experience_id is not None:
            self._current_exp = experience_id

        # Calculate MAE for this batch
        mae_value = torch.abs(predicted_y - true_y).mean().item()
        
        # Update global MAE
        self._global_mae.update(mae_value, len(true_y))
        
        # Update experience-specific MAE if we have a current experience
        if self._current_exp is not None:
            self._exp_mae[self._current_exp].update(mae_value, len(true_y))

    def result(self) -> Union[float, Dict[int, float]]:
        """
        Retrieves the running MAE based on emission mode.
        
        :return: Global MAE if emit_at="stream", current experience MAE if emit_at="experience"
        """
        if self._emit_at == "stream":
            return self._global_mae.result()
        else:  # "experience"
            if self._current_exp is not None and self._current_exp in self._exp_mae:
                return self._exp_mae[self._current_exp].result()
            return 0.0
            
    def exp_result(self, exp_id: Optional[int] = None) -> Union[float, Dict[int, float]]:
        """
        Get MAE for a specific experience or all experiences.
        
        :param exp_id: If provided, return MAE for that experience. 
                      If None, return dict of all experiences.
        :return: MAE for requested experience(s)
        """
        if exp_id is not None:
            if exp_id in self._exp_mae:
                return self._exp_mae[exp_id].result()
            return 0.0
        else:
            return {k: v.result() for k, v in self._exp_mae.items()}

    def reset(self, experience_id: Optional[int] = None) -> None:
        """
        Resets the metric.
        
        :param experience_id: If provided, reset only that experience's metrics.
                             If None, reset everything.
        :return: None.
        """
        if experience_id is not None:
            if experience_id in self._exp_mae:
                self._exp_mae[experience_id].reset()
        else:
            self._global_mae.reset()
            self._exp_mae = defaultdict(Mean)
            self._current_exp = None
            
__all__ = [
    "MeanAbsoluteError",
    "TaskAwareMeanAbsoluteError",
    "ExperienceAwareMeanAbsoluteError"
]

