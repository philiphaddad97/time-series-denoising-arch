import json
import numpy as np
from collections import namedtuple
from typing import List, Tuple, Any


def load_json(json_file: str) -> dict:
    """
    Load json file.
    :param json_file: json file to load
    :return: json file
    """
    with open(json_file, 'r') as f:
        return json.load(f)


def flatten_list(list_of_lists: List[List[Any]]) -> List[Any]:
    """
    Flatten a list of lists.
    :param list_of_lists: list of lists
    :return: flattened list
    """
    if list_of_lists == []:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten_list(list_of_lists[0]) + flatten_list(list_of_lists[1:])
    return list_of_lists[:1] + flatten_list(list_of_lists[1:])


def flatten_list_of_np_arrays(list_of_np_arrays: List[np.ndarray]) -> np.ndarray:
    """
    Flatten a list of numpy arrays.
    :param list_of_np_arrays: list of numpy arrays
    :return: flattened np array
    """
    return np.concatenate(list_of_np_arrays).ravel()


def get_data_stats_single(data: np.array) -> namedtuple:
    """
    Get statistics for data.
    :param data: data to get statistics for
    :return: namedtuple with statistics
    """
    data_stats = namedtuple('data_stats', ['median', 'mean', 'std', 'min', 'max'])
    return data_stats(np.median(data), np.mean(data), np.std(data), np.min(data), np.max(data))


def get_data_stats_list(data: List[np.array]) -> namedtuple:
    """
    Get statistics for data.
    :param data: data to get statistics for
    :return: namedtuple with statistics
    """
    f_data = flatten_list_of_np_arrays(data)
    return get_data_stats_single(f_data)


def get_num_samples_from_ts(ts: np.ndarray) -> int:
    """
    Get number of samples from time series.
    :param ts: time series
    :return: length of time series
    """
    return len(ts)


def compose(*functions):
    """
    Compose functions.
    :param functions: the functions to compose
    :return: a function that is the composition of the given functions
    """

    def inner(data, funcs=functions):
        result = data
        for f in reversed(funcs):
            result = f(result)
        return result

    return inner
