import torch
from torch import nn
from torchmetrics import Accuracy, Precision, Recall, F1Score, MetricCollection
from torch.nn import functional as F
import pytorch_lightning as pl
from utils.layers import TimeDistributed
from utils.metrics import AdjustedF1Score
from utils.anomaly_detectors import static_thresholding_detector, adaptive_threshold_detector
from typing import List, Tuple, Union, Literal
from functools import partial


class Encoder(pl.LightningModule):
    def __init__(self, input_shape, layers, dropout):
        super().__init__()
        self.lstms = nn.ModuleList()
        self.lstms.append(nn.LSTM(input_size=input_shape[2],
                                  hidden_size=layers[0],
                                  num_layers=1,
                                  batch_first=True))
        self.lstms.append(nn.Dropout(dropout))
        for i in range(1, len(layers)):
            self.lstms.append(nn.LSTM(input_size=layers[i - 1],
                                      hidden_size=layers[i],
                                      num_layers=1,
                                      batch_first=True))
            self.lstms.append(nn.Dropout(dropout))

    def forward(self, x):
        # take into account the dropout layers
        for i in range(0, len(self.lstms), 2):
            x, _ = self.lstms[i](x)
            x = self.lstms[i + 1](x)
        return x


class Decoder(pl.LightningModule):
    def __init__(self, layers, dropout):
        super().__init__()
        reversed_layers = layers[::-1]
        self.lstms = nn.ModuleList()
        self.lstms.append(nn.LSTM(input_size=reversed_layers[0],
                                  hidden_size=reversed_layers[0],
                                  num_layers=1,
                                  batch_first=True))
        self.lstms.append(nn.Dropout(dropout))
        for i in range(1, len(reversed_layers)):
            self.lstms.append(nn.LSTM(input_size=reversed_layers[i - 1],
                                      hidden_size=reversed_layers[i],
                                      num_layers=1,
                                      batch_first=True))
            self.lstms.append(nn.Dropout(dropout))

    def forward(self, x):
        # take into account the dropout layers
        for i in range(0, len(self.lstms), 2):
            x, _ = self.lstms[i](x)
            x = self.lstms[i + 1](x)
        return x


class LSTMAutoencoder(pl.LightningModule):
    """
    LSTM Autoencoder PyTorch Lightning Class
    """

    def __init__(
            self,
            input_shape: Tuple,
            layers: List,
            dropout: float = 0.2,
            anomaly_threshold: Union[float, Literal["adaptive_threshold_detector"]] = 0.99,
            loss_fn: str = "mse_loss",
            optimizer: str = "Adam",
            sync_dist: bool = False,
    ):
        super(LSTMAutoencoder, self).__init__()

        # use static thresholding if the threshold is a float
        if isinstance(anomaly_threshold, float):
           self.anomaly_threshold_detector = partial(static_thresholding_detector, threshold=anomaly_threshold)
        elif anomaly_threshold == "adaptive_threshold_detector":
           self.anomaly_threshold_detector = partial(adaptive_threshold_detector)

        self.input_shape = input_shape
        self.layers = layers
        self.dropout = dropout
        self.loss_fn = getattr(F, loss_fn)
        self.reconstruction_loss_fn = getattr(F, "l1_loss")
        self.optimizer = optimizer
        self.sync_dist = sync_dist
        metrics = MetricCollection([Accuracy(), Precision(), Recall(), F1Score(), AdjustedF1Score()])
        self.test_metrics = metrics.clone(prefix="test_")

        # Encoder
        self.encoder = Encoder(self.input_shape, self.layers, self.dropout)

        # Decoder
        self.decoder = Decoder(self.layers, self.dropout)

        # Time Distributed Dense
        self.td = TimeDistributed(nn.Linear(in_features=self.layers[0], out_features=self.input_shape[2]))

    def forward(self, x):
        # Encoder
        x = self.encoder(x)

        # Repeat Vector
        x_dims = len(x.shape)
        repeat_shape = [1] * x_dims
        repeat_shape[1] = self.input_shape[1]
        encoder_output = x[:, -1, :].clone().unsqueeze(1).requires_grad_(True)
        x = encoder_output.repeat(repeat_shape)

        # Decoder
        x = self.decoder(x)

        # Time Distributed Dense
        x = self.td(x)

        return x

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer)
        return optimizer(self.parameters())

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = self.loss_fn(x, x_hat)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        valid_loss = self.loss_fn(x, x_hat)
        self.log('valid_loss', valid_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
        return valid_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        all_loss = self.reconstruction_loss_fn(x_hat, x, reduction='none')
        row_wise_loss = all_loss.mean(dim=2)
        return row_wise_loss, y

    def test_epoch_end(self, outputs):
        row_wise_loss, y = zip(*outputs)
        # concat all the values on row_wise_loss in a single 1D tensor
        self.y_pred = self.anomaly_threshold_detector(row_wise_loss)
        for y_pred_item, y_item in zip(self.y_pred, y):
            y_pred_item = y_pred_item.to(dtype=torch.float64, device=self.device)
            y_item = y_item.to(dtype=torch.int64, device=self.device)
            self.test_metrics.update(y_pred_item, y_item)
        total_test_metrics = self.test_metrics.compute()
        self.log_dict(total_test_metrics, sync_dist=self.sync_dist)