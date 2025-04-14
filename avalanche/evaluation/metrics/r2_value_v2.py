################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 14-12-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################
from matplotlib.figure import Figure
import seaborn as sns
from numpy import arange
from typing import (
    Any,
    Callable,
    Iterable,
    Union,
    Optional,
    TYPE_CHECKING,
    List,
    Literal,
)

import wandb
import numpy as np
import torch
from PIL.Image import Image
from torch import Tensor
from torch.nn.functional import pad

from avalanche.benchmarks import NCScenario
from avalanche.evaluation import PluginMetric, Metric
from avalanche.evaluation.metric_results import (
    AlternativeValues,
    MetricValue,
    MetricResult,
)
from avalanche.evaluation.metric_utils import (
    default_cm_image_creator,
    phase_and_task,
    stream_type,
)

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class R2_value(Metric[Tensor]):
    """
    The standalone R² metric.

    Instances of this metric keep track of the R² value by receiving a
    pair of "ground truth" and "prediction" Tensors describing the labels of a
    minibatch.
    """

    def __init__(self):
        """
        Creates an instance of the standalone R² metric.

        By default, this metric in its initial state will return 0.
        The metric can be updated by using the `update` method, while the running
        R² value can be retrieved using the `result` method.
        """
        self._true_values = []
        self._predicted_values = []

    @torch.no_grad()
    def update(self, true_y: Tensor, predicted_y: Tensor) -> None:
        """
        Update the running R² value given the true and predicted labels.

        :param true_y: The ground truth values.
        :param predicted_y: The predicted values.
        :return: None.
        """
        self._true_values.append(true_y.cpu().numpy())
        self._predicted_values.append(predicted_y.cpu().numpy())

    def result(self) -> float:
        """
        Retrieves the R² value.

        Calling this method will not change the internal state of the metric.

        :return: The running R² value.
        """
        if not self._true_values or not self._predicted_values:
            return 0.0

        true_y = np.concatenate(self._true_values)
        predicted_y = np.concatenate(self._predicted_values)
        
        return r2_score(true_y, predicted_y)

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        self._true_values = []
        self._predicted_values = []



class StreamR2(PluginMetric[Tensor]):
    """
    The Stream R2 value metric.
    This plugin metric only works on the eval phase

    At the end of the eval phase, this metric logs the r2 value
    relative to all the patterns seen during eval.

    The metric can log either a Tensor or a PIL Image representing the
    r2 value.
    """

    def __init__(
        self,
        save_image: bool = True
    ):
        """
        Creates an instance of the Stream R2 metric.

        We recommend to set `save_image=False` if the runtime is too large.
        In fact, a large number of classes may increase the computation time
        of this metric.
        """
        super().__init__()
        self.save_image = save_image
        # self.r2_metric = R2_value()
        self.reset()

    def reset(self) -> None:
        self.true_values = []
        self.predicted_values = []

    def result(self) -> Tensor:
        true_y = np.concatenate(self.true_values)
        predicted_y = np.concatenate(self.predicted_values)
        
        return r2_score(true_y, predicted_y)

    def update(self, true_y: Tensor, predicted_y: Tensor) -> None: #find how to access targets from true_y, predicted_y
        self.true_values.append(true_y.cpu().numpy())
        self.predicted_values.append(predicted_y.cpu().numpy())

    def before_eval(self, strategy) -> None:
        self.reset()

    def after_eval_iteration(self, strategy: "SupervisedTemplate") -> None:
        super().after_eval_iteration(strategy)
        self.update(strategy.mb_y, strategy.mb_output)

    def after_eval(self, strategy: "SupervisedTemplate") -> MetricResult:
        return self._package_result(strategy)

    def _package_result(self, strategy: "SupervisedTemplate") -> MetricResult:
        exp_r2 = self.result()
        phase_name, _ = phase_and_task(strategy)
        stream = stream_type(strategy.experience)
        e_num = strategy.experience.origin_stream if hasattr(strategy.experience, 'origin_stream') else strategy.experience.current_experience
        metric_name = "{}/{}_phase/{}_stream".format(str(self), phase_name, stream)
        plot_x_position = strategy.clock.train_iterations

        if self.save_image:
            # fig, ax = plt.subplots()
            fig, (ax, ax_true, ax_pred) = plt.subplots(1, 3, figsize=(18, 6))
            true_y = np.concatenate(self.true_values)
            predicted_y = np.concatenate(self.predicted_values)
            ax.scatter(true_y, predicted_y, alpha=0.5)
            
            # ax.plot([true_y.min(), true_y.max()], [true_y.min(), true_y.max()], 'k--', lw=2)
            model = LinearRegression()
            true_y_reshaped = true_y.reshape(-1, 1)
            pred_y_reshaped = predicted_y.reshape(-1, 1)
            model.fit(true_y_reshaped, predicted_y)
            predicted_fit = model.predict(true_y_reshaped)
            slope = model.coef_[0].item()
            ax.plot(true_y, predicted_fit, color='red', linewidth=2, label=f'Slope: {slope:.2f}')

            # Annotate with R² value
            r2_value = r2_score(true_y, predicted_y)
            # ax.text(0.05, 0.95, f'R² = {r2_value:.2f}', transform=ax.transAxes, fontsize=14,
            #         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            ax.text(0.05, 0.95, f'R² = {r2_value:.2f}', transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            ax.set_xlabel('True Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'Stream Regression Plot')
            ax.legend()

            # Distribution of true values
            # scaled_true_y = np.ravel(1000 * true_y)
            # scaled_predicted_y = np.ravel(1000 * predicted_y)
      
            scaled_true_y = np.ravel(true_y)
            scaled_predicted_y = np.ravel(predicted_y)

            # Create a DataFrame for the KDE plot
            data_true = pd.DataFrame({'Value': scaled_true_y, 'Label Type': 'True'})
            data_pred = pd.DataFrame({'Value': scaled_predicted_y, 'Label Type': 'Predicted'})
            data = pd.concat([data_true, data_pred])            
            
            sns.kdeplot(scaled_true_y, ax=ax_true, color='blue', fill=True, common_norm=False, common_grid=True, bw_adjust=0.5)
            ax_true.set_title('True Values Distribution')
            ax_true.set_xlabel('Scaled Value')
            ax_true.set_ylabel('Density')

            # Distribution of predicted values
            sns.kdeplot(data=data, x='Value', hue='Label Type', ax=ax_pred, fill=True, common_norm=False, common_grid=True, bw_adjust=0.5)
            ax_pred.set_title('Predicted Values Distribution')
            ax_pred.set_xlabel('Scaled Value')
            ax_pred.set_ylabel('Density')

            
            max_val = true_y.max()
            min_val = true_y.min()
            ax.set_xlim([min_val, max_val])
            ax.set_ylim([min_val, max_val])
            

            
            metric_representation = MetricValue(
                self,
                metric_name,
                r2_value,
                plot_x_position,
            )
        else:
            metric_representation = MetricValue(
                self, metric_name, exp_r2, plot_x_position
            )

        return [metric_representation]

    def __str__(self):
        return "R2_Stream"


def r2_metrics(
    save_image=False,
    stream=False,
) -> List[PluginMetric]:
    """Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param save_image: If True, a graphical representation of the truth and predicted values 
    plotted against each other will be plotted, too. If False, only the Tensor representation
        will be logged. Defaults to True.
    :param stream: If True, will return a metric able to log
        the r2 averaged over the entire evaluation stream
        of experiences.

    :return: A list of plugin metrics.
    """

    metrics: List[PluginMetric] = []

    if stream:
        metrics.append(
            StreamR2(save_image=save_image)
        )

    return metrics


__all__ = [
    "StreamR2",
    "r2_metrics",
]
