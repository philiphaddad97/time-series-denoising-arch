import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from functools import partial
from utils import load_json, compose
from torch.utils.data import DataLoader
from datasets_classes import UCRDataset
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Union, Callable, Optional


class UCRDataModule(pl.LightningDataModule):

    def __init__(self,
                 dataset_file_path: str,
                 dataset_configs_path: str = 'datasets/datasets_configs/ucr_config.json',
                 batch_size: int = 32,
                 seq_len: int = 128,
                 step: int = 1,
                 valid_size: float = 0.1,  # percentage of the training set to use as validation set
                 shuffle: bool = False,
                 normalize: bool = True,
                 normalize_range: Tuple[float, float] = (-1, 1),
                 num_workers: int = 0):
        super().__init__()
        self.dataset_file_path = dataset_file_path
        self.dataset_config = load_json(dataset_configs_path)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.step = step
        self.valid_size = valid_size
        self.shuffle = shuffle
        self.normalize = normalize
        self.normalization_range = normalize_range
        self.num_workers = num_workers
        self.dataset_values = None
        self.dataset_labels = None
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.test_labels = None
        self.dataset_name = None
        self.train_idx = None
        self.num_features = None
        self.df_reader_functions_map = {
            'type_1': partial(pd.read_csv, header=None),
            'type_2': compose(pd.DataFrame.transpose, partial(pd.read_csv, sep='\s+', header=None))
        }

    @property
    def name(self):
        return self.dataset_name

    def prepare_data(self) -> None:
        # extract file name from the path using pathlib
        file_name = os.path.basename(self.dataset_file_path)
        # remove the extension from the file name
        file_name = os.path.splitext(file_name)[0]
        # extract dataset characteristics from the file name
        file_name_elements = file_name.split('_')
        dataset_id = int(file_name_elements[0])
        self.dataset_name = file_name_elements[3]
        self.train_idx = int(file_name_elements[4])
        anomaly_start_index = int(file_name_elements[5])
        anomaly_end_index = int(file_name_elements[6])
        df_reader_function_type = 'type_2' \
            if dataset_id in self.dataset_config['main_config']['space_separated_datasets_ids'] else 'type_1'
        df_reader_function = self.df_reader_functions_map[df_reader_function_type]
        df_data = df_reader_function(self.dataset_file_path)
        self.dataset_values = df_data.values
        num_samples = self.dataset_values.shape[0]
        labels = np.zeros(num_samples)
        labels[anomaly_start_index:anomaly_end_index] = 1
        self.dataset_labels = labels.copy().astype(np.int64)
        # import pdb; pdb.set_trace()

    def setup(self, stage: Optional[str] = None) -> None:
        valid_idx = int(self.train_idx * (1 - self.valid_size))
        self.train_data = self.dataset_values[:valid_idx]
        self.valid_data = self.dataset_values[valid_idx:self.train_idx]
        self.test_data = self.dataset_values[self.train_idx:]
        self.test_labels = self.dataset_labels[self.train_idx:]

        # normalize data
        if self.normalize:
            scaler = MinMaxScaler(feature_range=self.normalization_range)
            self.train_data = scaler.fit_transform(self.train_data)
            self.valid_data = scaler.transform(self.valid_data)
            self.test_data = scaler.transform(self.test_data)

        # squeeze to 1D
        self.train_data = np.squeeze(self.train_data)
        self.valid_data = np.squeeze(self.valid_data)
        self.test_data = np.squeeze(self.test_data)

    def train_dataloader(self):
        dataset = UCRDataset(data=self.train_data,
                             labels=None,
                             seq_len=self.seq_len,
                             step=self.step,
                             mode='train')

        return DataLoader(dataset=dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def val_dataloader(self):
        dataset = UCRDataset(data=self.valid_data,
                             labels=None,
                             seq_len=self.seq_len,
                             step=self.step,
                             mode='valid')

        return DataLoader(dataset=dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=self.shuffle)

    def test_dataloader(self):
        dataset = UCRDataset(data=self.test_data,
                             labels=self.test_labels,
                             seq_len=self.seq_len,
                             step=self.step,
                             mode='test')

        return DataLoader(dataset=dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)
