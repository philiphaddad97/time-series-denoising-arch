import math
import torch
from torch import Tensor
from typing import List, Tuple, Union
from scipy import stats
import numpy as np
from .utlis import mitigate_false_positive


def static_thresholding_detector(
    data: Union[List[Tensor], Tuple[Tensor]], threshold: float
) -> Tuple[Tensor]:
    combined_data = torch.cat(data, dim=0)
    anomaly_threshold = torch.quantile(combined_data, threshold)
    y_pred = tuple(
        torch.where(data_row > anomaly_threshold, 1.0, 0.0) for data_row in data
    )
    return y_pred


def adaptive_threshold_detector(
    data: Union[List[Tensor], Tuple[Tensor]],
    window_size_factor: int = 10,
    step_size_factor: float = 3 * 10,
    z_value: int = 3,
    change_threshold: float = 0.1,
) -> Tuple[Tensor]:
    """Adaptive Threshold Detector
    1. combined the data
    2. calculate z-score
    3. process the data window by window
    """
    data_1D = torch.cat(data, dim=0).ravel()
    anomaly_score = stats.zscore(data_1D.detach().cpu().numpy(), nan_policy="omit")

    window_size = len(anomaly_score) // window_size_factor
    step_size = len(anomaly_score) // step_size_factor

    y_pred = np.zeros(len(anomaly_score))

    for i in range(0, len(anomaly_score) - window_size, step_size):
        window = anomaly_score[i : i + window_size]
        window_mean = np.mean(window)
        window_std = np.std(window)

        for j, value in enumerate(window):
            if (
                (window_mean - z_value * window_std)
                < value
                < (window_mean + z_value * window_std)
            ):
                y_pred[i + j] = 0.0
            else:
                y_pred[i + j] = 1.0

    y_pred = mitigate_false_positive(y_pred, anomaly_score, change_threshold)

    # Restore the data to the same structre before starting the process
    y_pred = tuple(
        torch.from_numpy(
            y_pred[: -math.prod(data[-1].shape)].reshape(len(data) - 1, *data[0].shape)
        )
    ) + tuple(
        torch.from_numpy(
            # The last batch is handled separately (example: batch size = 16 but we got only 12 samples)
            y_pred[-math.prod(data[-1].shape) :].reshape(1, *data[-1].shape)
        )
    )

    return y_pred
