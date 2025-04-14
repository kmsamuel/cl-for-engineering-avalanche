################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 30-12-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from typing import List, Optional, Union, Dict

import torch
from torch import Tensor
import matplotlib as plt
import numpy as np
from sklearn.metrics import r2_score
from avalanche.evaluation import Metric, GenericPluginMetric
from avalanche.evaluation.metrics.mean import Mean
from collections import defaultdict

class RegressionMetrics(Metric[float]):
    """Accuracy metric. This is a standalone metric.

    The update method computes the accuracy incrementally
    by keeping a running average of the <prediction, target> pairs
    of Tensors provided over time.

    The "prediction" and "target" tensors contain regression values.

    Each time `result` is called, this metric emits the average accuracy
    across all predictions made since the last `reset`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an accuracy value of 0.
    """

    def __init__(self):
        """Creates an instance of the standalone Accuracy metric.

        By default this metric in its initial state will return an accuracy
        value of 0. The metric can be updated by using the `update` method
        while the running accuracy can be retrieved using the `result` method.
        """
        self._mean_accuracy = Mean()
        """The mean utility that will be used to store the running accuracy."""

    @torch.no_grad()
    def update(
        self,
        predicted_y: Tensor,
        true_y: Tensor,
    ) -> None:
        """Update the running accuracy given the true and predicted labels.

        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.

        :return: None.
        """
        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)
        
        self.true_y = true_y
        self.predicted_y = predicted_y

        # if len(true_y) != len(predicted_y):
        #     raise ValueError("Size mismatch for true_y and predicted_y tensors")

        # # Check if logits or labels
        # if len(predicted_y.shape) > 1:
        #     # Logits -> transform to labels
        #     predicted_y = torch.max(predicted_y, 1)[1]

        # if len(true_y.shape) > 1:
        #     # Logits -> transform to labels
        #     true_y = torch.max(true_y, 1)[1]
        # AE = torch.abs(predicted_y - true_y)
        MAE = torch.mean(torch.abs(predicted_y - true_y))

        # SE = torch.mean((predicted_y - true_y) ** 2)

        # RMSE = torch.sqrt(torch.mean((predicted_y - true_y) ** 2))

        # true_positives = float(torch.sum(torch.eq(predicted_y, true_y)))
        total_patterns = len(true_y)
        # self._mean_accuracy.update(true_positives / total_patterns, total_patterns)
        # self._mean_accuracy.update(SE, total_patterns)
        self._mean_accuracy.update(MAE.item(), total_patterns)


    def result(self) -> float:
        """Retrieves the running accuracy.

        Calling this method will not change the internal state of the metric.

        :return: The current running accuracy, which is a float value
            between 0 and 1.
        """
        return self._mean_accuracy.result(), 

    def reset(self) -> None:
        """Resets the metric.

        :return: None.
        """
        self._mean_accuracy.reset()
    


class RegressionPluginMetric(GenericPluginMetric[float, RegressionMetrics]):
    """
    Base class for all accuracies plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode, split_by_task=False):
        """Creates the Accuracy plugin

        :param reset_at:
        :param emit_at:
        :param mode:
        :param split_by_task: whether to compute task-aware accuracy or not.
        """
        super().__init__(RegressionMetrics(), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._metric.update(strategy.mb_output, strategy.mb_y)

class MinibatchReg(RegressionPluginMetric):
    """
    The minibatch plugin accuracy metric.
    This metric only works at training time.

    This metric computes the average accuracy over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochAccuracy` instead.
    """

    def __init__(self):
        """
        Creates an instance of the MinibatchAccuracy metric.
        """
        super(MinibatchReg, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train"
        )

    def __str__(self):
        # return "MSE_MB"
        return "MAE_MB"


class EpochReg(RegressionPluginMetric):
    """
    The average accuracy over a single training epoch.
    This plugin metric only works at training time.

    The accuracy will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochAccuracy metric.
        """

        super(EpochReg, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train"
        )

    def __str__(self):
        # return "MSE_Epoch"
        return "MAE_Epoch"

class RunningEpochReg(RegressionPluginMetric):
    """
    The average accuracy across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the accuracy averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self):
        """
        Creates an instance of the RunningEpochAccuracy metric.
        """

        super(RunningEpochReg, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train"
        )

    def __str__(self):
        # return "MSE_RunningAcc_Epoch"
        return "MAE_RunningAcc_Epoch"


class ExperienceReg(RegressionPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average accuracy over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceAccuracy metric
        """
        super(ExperienceReg, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )
        # self.true_y = []
        # self.predicted_y = []

    def __str__(self):
        # return "MSE_Exp"
        return "MAE_Exp"

    # def update(self, strategy):
    #     """
    #     Update the metric with new true and predicted values.
    #     """
    #     self.true_y.extend(strategy.mb_y)
    #     self.predicted_y.extend(strategy.mb_output)
    
    # def compute(self):
    #     """
    #     Compute the R² and plot true vs predicted values.
    #     """
    #     # Ensure true_y and predicted_y are not empty
    #     if not self.true_y or not self.predicted_y:
    #         return None

    #     else:
    #         true_y = np.array(self.true_y)
    #         predicted_y = np.array(self.predicted_y)

    #         # Calculate R²
    #         r2 = r2_score(true_y, predicted_y)

    #         return r2
    
    # def plot(self):
    #     if not self.true_y or not self.predicted_y:
    #         print('********************PRINTING NONE****************************************')

    #         return None
    #     else:
    #         true_y = np.array(self.true_y)
    #         predicted_y = np.array(self.predicted_y)
            
    #         print('************************************************************')
    #         print('true_y:', true_y)
    #         print('pred_y:', predicted_y)

    #         # Calculate R²
    #         r2 = r2_score(true_y, predicted_y)
    #         # Plot true vs. predicted values
    #         plt.figure(figsize=(10, 6))
    #         plt.scatter(true_y, predicted_y, alpha=0.6, edgecolors='w', linewidth=0.5)
    #         plt.plot([true_y.min(), true_y.max()], [true_y.min(), true_y.max()], 'r--', lw=2)
    #         plt.xlabel('True Values')
    #         plt.ylabel('Predicted Values')
    #         plt.title(f'True vs Predicted Values\nR² = {r2:.2f}')
    #         plt.show()

class StreamReg(RegressionPluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average accuracy over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of StreamAccuracy metric
        """
        super(StreamReg, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        # return "MSE_Stream"
        return "MAE_Stream"

        
    
class TrainedExperienceReg(RegressionPluginMetric):
    """
    At the end of each experience, this plugin metric reports the average
    accuracy for only the experiences that the model has been trained on so far.

    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of TrainedExperienceAccuracy metric by first
        constructing AccuracyPluginMetric
        """
        super(TrainedExperienceReg, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )
        self._current_experience = 0

    def after_training_exp(self, strategy):
        self._current_experience = strategy.experience.current_experience
        # Reset average after learning from a new experience
        self.reset()
        return super().after_training_exp(strategy)

    def update(self, strategy):
        """
        Only update the accuracy with results from experiences that have been
        trained on
        """
        if strategy.experience.current_experience <= self._current_experience:
            RegressionPluginMetric.update(self, strategy)

    def __str__(self):
        # return "MSE_On_Trained_Experiences"
        return "MAE_On_Trained_Experiences"


def regression_metrics(
    *,
    minibatch=False,
    epoch=False,
    epoch_running=False,
    experience=False,
    stream=False,
    trained_experience=False,
) -> List[RegressionPluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param minibatch: If True, will return a metric able to log
        the minibatch accuracy at training time.
    :param epoch: If True, will return a metric able to log
        the epoch accuracy at training time.
    :param epoch_running: If True, will return a metric able to log
        the running epoch accuracy at training time.
    :param experience: If True, will return a metric able to log
        the accuracy on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the accuracy averaged over the entire evaluation stream of experiences.
    :param trained_experience: If True, will return a metric able to log
        the average evaluation accuracy only for experiences that the
        model has been trained on

    :return: A list of plugin metrics.
    """

    metrics: List[RegressionPluginMetric] = []
    if minibatch:
        metrics.append(MinibatchReg())

    if epoch:
        metrics.append(EpochReg())

    if epoch_running:
        metrics.append(RunningEpochReg())

    if experience:
        metrics.append(ExperienceReg())
        # r2 = ExperienceReg().compute()
        # print(f'R2 value: {r2:.2f}')
        # ExperienceReg().plot()

    if stream:
        metrics.append(StreamReg())

    if trained_experience:
        metrics.append(TrainedExperienceReg())

    return metrics

__all__ = [
    "RegressionMetrics",
    "MinibatchReg",
    "EpochReg",
    "RunningEpochReg",
    "ExperienceReg",
    "StreamReg",
    "TrainedExperienceReg",
    "regression_metrics",
]
