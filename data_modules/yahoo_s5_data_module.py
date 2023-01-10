import os
import re
import random
import pandas as pd
import pytorch_lightning as pl
from utils import load_json
from utils.normalization_functions import normalize_split_data
from torch.utils.data import DataLoader
from datasets_classes import YahooS5Dataset
from typing import Tuple, Optional


class YahooS5DataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: str,
                 dataset_name: str,
                 dataset_configs_path: str,
                 train_size: float = 0.8,
                 valid_size: float = 0.1,
                 test_size: float = 0.1,
                 batch_size: int = 32,
                 seq_len: int = 128,
                 step: int = 1,
                 shuffle: bool = False,
                 normalize: bool = True,
                 normalize_range: Tuple[int, int] = (-1, 1),
                 num_workers: int = 0):
        super().__init__()
        self.test_labels = None
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.dataset_config = load_json(dataset_configs_path)[self.dataset_name]
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.step = step
        self.shuffle = shuffle
        self.normalize = normalize
        self.normalization_range = normalize_range
        self.num_workers = num_workers
        self.splitting_functions_map = {
            'end_of_each_file': self.__end_of_each_file_splitting,
        }
        self.dataset_values = None
        self.dataset_labels = None
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.test_labels = None

    def __end_of_each_file_splitting(self):
        self.train_data = [d[:int(self.train_size * len(d))] for d in self.dataset_values]
        self.valid_data = [d[int(self.train_size * len(d)):int((self.train_size + self.valid_size) * len(d))]
                           for d in self.dataset_values]
        self.test_data = [d[int((self.train_size + self.valid_size) * len(d)):] for d in self.dataset_values]
        self.test_labels = [d[int((self.train_size + self.valid_size) * len(d)):] for d in self.dataset_labels]

    def prepare_data(self) -> None:
        value_column = self.dataset_config['value_column']
        label_column = self.dataset_config['label_column']
        file_name_pattern = re.compile(self.dataset_config['file_name_pattern'])
        dataset_files = [os.path.join(self.dataset_path, file) for file in os.listdir(self.dataset_path) if
                         file_name_pattern.match(file)]
        files_dfs = [pd.read_csv(dataset_file) for dataset_file in dataset_files]
        self.dataset_values = [df[value_column].values for df in files_dfs]
        self.dataset_labels = [df[label_column].values for df in files_dfs]

    def setup(self, splitting_strategy: str = 'end_of_each_file', stage: Optional[str] = None) -> None:
        # split the dataset into train, valid and test
        self.splitting_functions_map[splitting_strategy]()

        # normalize the dataset
        if self.normalize:
            self.train_data, self.valid_data, self.test_data = normalize_split_data(self.train_data,
                                                                                    self.valid_data,
                                                                                    self.test_data,
                                                                                    self.normalization_range)

    def train_dataloader(self):
        if self.shuffle:
            random.shuffle(self.train_data)
        dataset = YahooS5Dataset(data=self.train_data,
                                 labels=None,
                                 seq_len=self.seq_len,
                                 step=self.step,
                                 mode='train')
        return DataLoader(dataset=dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def val_dataloader(self):
        dataset = YahooS5Dataset(data=self.valid_data,
                                 labels=None,
                                 seq_len=self.seq_len,
                                 step=self.step,
                                 mode='valid')
        return DataLoader(dataset=dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        dataset = YahooS5Dataset(data=self.test_data,
                                 labels=self.test_labels,
                                 seq_len=self.seq_len,
                                 step=self.step,
                                 mode='test')
        return DataLoader(dataset=dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)
