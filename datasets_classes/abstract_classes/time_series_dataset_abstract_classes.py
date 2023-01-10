import torch
import numpy as np
from bisect import bisect
from itertools import accumulate
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import List, Literal, Union
from utils import get_num_samples_from_ts


class TimeSeriesDataset(Dataset, ABC):
    # make the init accept multiple types of data
    def __init__(self, data: Union[List[np.ndarray], np.ndarray],
                 labels: Union[List[np.ndarray], np.ndarray, None],
                 seq_len: int,
                 step: int,
                 mode: Literal['train', 'valid', 'test']):
        super().__init__()
        self._data = data
        self._labels = labels
        self._seq_len = seq_len
        self._step = step
        self._mode = mode
        self._n = 0
        self._get_sample_fun_map = {
            'train': self._get_sample_1,
            'valid': self._get_sample_1,
            'test': self._get_sample_2
        }
        self.__get_sample = self._get_sample_fun_map[self._mode]

    @abstractmethod
    def _go_to_idx(self, idx: int):
        pass

    @abstractmethod
    def _get_sample_1(self):
        pass

    @abstractmethod
    def _get_sample_2(self):
        pass

    def _get_num_of_sequences_from_ts(self, ts: np.ndarray) -> int:
        num_samples_ts = get_num_samples_from_ts(ts)
        possible_steps = list(range(num_samples_ts - self._seq_len, 0, -self._step))
        possible_ends = [x + self._seq_len for x in possible_steps]
        valid_ends = list(filter(lambda x: x < num_samples_ts, possible_ends))
        num_valid_ends = len(valid_ends)
        return num_valid_ends

    def __getitem__(self, idx):
        self._go_to_idx(idx)
        sample = self.__get_sample()
        return sample


class TimeSeriesDatasetMultipleFiles(TimeSeriesDataset, ABC):
    def __init__(self, data: List[np.ndarray],
                 labels: Union[List[np.ndarray], None],
                 seq_len: int,
                 step: int,
                 mode: Literal['train', 'valid', 'test']):
        super().__init__(data=data,
                         labels=labels,
                         seq_len=seq_len,
                         step=step,
                         mode=mode)
        self._current_file = 0
        self._num_samples_per_file = self._get_num_samples_per_file()
        self._num_sequences_per_file = self._get_num_sequences_per_file()
        self._accumulated_num_sequences_per_file = list(accumulate(self._num_sequences_per_file))

    def __len__(self):
        return sum(self._num_sequences_per_file)

    def _get_num_samples_per_file(self) -> List[int]:
        return [get_num_samples_from_ts(ts) for ts in self._data]

    def _get_num_sequences_per_file(self) -> List[int]:
        return [self._get_num_of_sequences_from_ts(ts) for ts in self._data]

    def _go_to_idx(self, idx: int):
        if idx >= self.__len__():
            raise IndexError("Index out of range")
        if idx in self._accumulated_num_sequences_per_file:
            self._current_file = self._accumulated_num_sequences_per_file.index(idx)
            self._n = 0
        else:
            file_num = bisect(self._accumulated_num_sequences_per_file, idx)
            self._current_file = file_num
            self._n = idx - self._accumulated_num_sequences_per_file[file_num - 1] if file_num != 0 else idx

    def _get_sample_1(self):
        sample = self._data[self._current_file][self._n*self._step: self._n*self._step + self._seq_len]
        sample = np.expand_dims(sample, axis=1)
        return torch.from_numpy(sample).float()

    def _get_sample_2(self):
        sample = self._data[self._current_file][self._n*self._step: self._n*self._step + self._seq_len]
        label = self._labels[self._current_file][self._n*self._step: self._n*self._step+ self._seq_len]
        sample = np.expand_dims(sample, axis=1)
        return torch.from_numpy(sample).float(), label


class TimeSeriesDatasetSingleFile(TimeSeriesDataset, ABC):
    def __init__(self, data: np.ndarray,
                 labels: Union[np.ndarray, None],
                 seq_len: int,
                 step: int,
                 mode: Literal['train', 'valid', 'test']):
        super().__init__(data=data,
                         labels=labels,
                         seq_len=seq_len,
                         step=step,
                         mode=mode)
        self._num_samples = self._get_num_samples()
        self._num_sequences = self._get_num_sequences()

    def __len__(self):
        return self._num_sequences

    def _get_num_samples(self) -> int:
        return get_num_samples_from_ts(self._data)

    def _get_num_sequences(self) -> int:
        return self._get_num_of_sequences_from_ts(self._data)

    def _go_to_idx(self, idx: int):
        if idx >= self.__len__():
            raise IndexError("Index out of range")
        self._n = idx

    def _get_sample_1(self):
        sample = self._data[self._n*self._step: self._n*self._step+ self._seq_len]
        sample = np.expand_dims(sample, axis=1)
        return torch.from_numpy(sample).float()

    def _get_sample_2(self):
        sample = self._data[self._n*self._step: self._n*self._step + self._seq_len]
        label = self._labels[self._n*self._step: self._n*self._step + self._seq_len]
        sample = np.expand_dims(sample, axis=1)
        return torch.from_numpy(sample).float(), label
