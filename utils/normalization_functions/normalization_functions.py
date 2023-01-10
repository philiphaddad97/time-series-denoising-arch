import numpy as np
from typing import List, Tuple, Dict
from utils import get_data_stats_single, get_data_stats_list, flatten_list_of_np_arrays
from sklearn.preprocessing import MinMaxScaler


def normalize_data_single(data: np.ndarray, normalize_range: Tuple[int, int] = (-1, 1)) -> np.ndarray:
    """
    Normalize data.
    :param normalize_range: range to normalize to
    :param data: data to normalize
    :return: normalized data
    """
    data_stats = get_data_stats_single(data)
    a = normalize_range[0]
    b = normalize_range[1]
    min_data = data_stats.min
    max_data = data_stats.max
    normalized_data = (data - min_data) / (max_data - min_data) * (b - a) + a
    return normalized_data


def normalize_data_all(data: List[np.array],
                       normalize_range: Tuple[int, int] = (-1, 1),
                       data_stats: Dict = None) -> List[np.ndarray]:
    if not data_stats:
        data_stats = get_data_stats_list(data)
    a = normalize_range[0]
    b = normalize_range[1]
    min_data = data_stats.min
    max_data = data_stats.max
    normalized_data = [(x - min_data) / (max_data - min_data) * (b - a) + a for x in data]
    return normalized_data


def normalize_data_list_one_per_time(data: List[np.ndarray],
                                     normalize_range: Tuple[int, int] = (-1, 1)) -> List[np.ndarray]:
    """
    Normalize data.
    :param normalize_range: range to normalize to
    :param data: data to normalize
    :return: normalized data
    """
    result = list()
    for d in data:
        d_stats = get_data_stats_single(d)
        a = normalize_range[0]
        b = normalize_range[1]
        result.append((d - d_stats.min) / (d_stats.max - d_stats.min) * (b - a) + a)
    return result


def normalize_split_data(train_data: List[np.array],
                         validation_data: List[np.array],
                         test_data: List[np.array],
                         normalize_range: Tuple[int, int]) -> Tuple[List[np.array], List[np.array], List[np.array]]:
    """
    Normalize data.
    :param normalize_range: range to normalize to
    :param train_data: train data
    :param validation_data: validation data
    :param test_data: test data
    :return: normalized data
    """
    train_data_combined = flatten_list_of_np_arrays(train_data)
    scaler = MinMaxScaler(feature_range=normalize_range)
    scaler.fit(train_data_combined.reshape(-1, 1))

    normalized_train_data = [scaler.transform(x.reshape(-1, 1)).reshape(-1) for x in train_data]
    normalized_validation_data = [scaler.transform(x.reshape(-1, 1)).reshape(-1) for x in validation_data]
    normalized_test_data = [scaler.transform(x.reshape(-1, 1)).reshape(-1) for x in test_data]

    return normalized_train_data, normalized_validation_data, normalized_test_data
