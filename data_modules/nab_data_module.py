import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from utils import load_json
from utils.nab.corpus import Corpus
from utils.nab.labelers import LabelCombiner, CorpusLabel
from torch.utils.data import DataLoader
from datasets_classes import NABDataset
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


class NABDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        train_size: float,
        valid_size: float,
        test_size: float,
        batch_size: int = 32,
        seq_len: int = 128,
        step: int = 128,
        shuffle: bool = False,
        normalize: bool = True,
        normalize_range: Tuple[float, float] = (-1, 1),
        num_workers: int = 0,
        nab_threshold: float = 0.5,
        nab_window_size: float = 0.1,
        probationary_percent: float = 0.15,
        verbosity: float = 0,
        dataset_configs_path: str = "datasets/datasets_configs/nab_config.json",
    ):

        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config = load_json(dataset_configs_path)
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
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
        self.dataset_name = dataset_name
        self.num_features = None
        self.nab_window_size = nab_window_size
        self.nab_threshold = nab_threshold
        self.nab_window_size = nab_window_size
        self.verbosity = verbosity
        self.probationary_percent = probationary_percent
        self.dataset_path = self.dataset_config["nab_path"]
        self.nab_raw_data_directory_path = os.path.join(self.dataset_path, "data")
        self.nab_label_directory_path = os.path.join(
            os.path.join(self.dataset_path, "labels"), "raw"
        )
        self.nab_combined_windows_path = os.path.join(
            os.path.join(self.dataset_path, "labels"), "combined_windows.json"
        )
        self.nab_combined_labels_path = os.path.join(
            os.path.join(self.dataset_path, "labels"), "combined_labels.json"
        )

    @property
    def name(self):
        return self.dataset_name

    def _get_labels(self):
        corpus = Corpus(self.nab_raw_data_directory_path)
        labelCombiner = LabelCombiner(
            self.nab_label_directory_path,
            corpus,
            threshold=self.nab_threshold,
            windowSize=self.nab_window_size,
            probationaryPercent=self.probationary_percent,
            verbosity=self.verbosity,
        )

        labelCombiner.combine()
        labelCombiner.write(
            self.nab_combined_labels_path, self.nab_combined_windows_path
        )
        corpusLabel = CorpusLabel(self.nab_combined_windows_path, corpus)
        corpusLabel.validateLabels()

        corpusLabel.getLabels()

        return (
            corpusLabel.labels[self.dataset_config["datasets"][self.dataset_name]]
            .values[:, 1]
            .astype("int64")
        )

    def prepare_data(self) -> None:
        df_data = pd.read_csv(
            os.path.join(
                self.nab_raw_data_directory_path,
                self.dataset_config["datasets"][self.dataset_name],
            )
        )
        self.dataset_values = np.array(df_data["value"])
        self.labels = self._get_labels()

    def setup(self) -> None:
        self.train_data = self.dataset_values[
            : int(self.train_size * len(self.dataset_values))
        ]
        self.valid_data = self.dataset_values[
            len(self.train_data) : int(self.valid_size * len(self.dataset_values))
            + len(self.train_data)
        ]
        self.test_data = self.dataset_values[
            len(self.train_data)
            + len(self.valid_data) : int(self.test_size * len(self.dataset_values))
            + len(self.train_data)
            + len(self.valid_data)
        ]
        self.test_labels = self.labels[
            len(self.labels) - len(self.test_data) : len(self.labels)
        ]
        # import pdb; pdb.set_trace()

        # normalize data
        if self.normalize:
            scaler = MinMaxScaler(feature_range=self.normalization_range)
            self.train_data = scaler.fit_transform(self.train_data.reshape(-1, 1))
            self.valid_data = scaler.transform(self.valid_data.reshape(-1, 1))
            self.test_data = scaler.transform(self.test_data.reshape(-1, 1))

        # squeeze to 1D
        self.train_data = np.squeeze(self.train_data)
        self.valid_data = np.squeeze(self.valid_data)
        self.test_data = np.squeeze(self.test_data)

    def train_dataloader(self):
        dataset = NABDataset(
            data=self.train_data,
            labels=None,
            seq_len=self.seq_len,
            step=self.step,
            mode="train",
        )

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def val_dataloader(self):
        dataset = NABDataset(
            data=self.valid_data,
            labels=None,
            seq_len=self.seq_len,
            step=self.step,
            mode="valid",
        )

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )

    def test_dataloader(self):
        dataset = NABDataset(
            data=self.test_data,
            labels=self.test_labels,
            seq_len=self.seq_len,
            step=self.step,
            mode="test",
        )

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
