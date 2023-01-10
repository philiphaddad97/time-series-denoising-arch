import argparse
from data_modules import YahooS5DataModule, NABDataModule, UCRDataModule
from models import LSTMAutoencoder
from pytorch_lightning import Trainer

# import torch callbacks: EarlyStopping, ModelCheckpoint, and LearningRateMonitor
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)

parser = argparse.ArgumentParser(description="Optional parameters: dataset, batch_size, seq_len, step")
parser.add_argument("--dataset", default="s5", required=False)
parser.add_argument("--batch_size", default="4", required=False)
parser.add_argument("--seq_len", default="24", required=False)
parser.add_argument("--step", default="1", required=False)
args = parser.parse_args()


def main():
    dataset = args.dataset
    batch_size = int(args.batch_size)
    seq_len = int(args.seq_len)
    step = int(args.step)

    if dataset == "ucr":
        data_module = UCRDataModule(
            dataset_file_path="datasets/datasets_files/ucr_anomaly/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt",
            seq_len=seq_len,
            batch_size=batch_size,
            step=step
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="valid_loss",
            dirpath="checkpoints/",
            filename="ucr_anmaly-{epoch:02d}-{val_loss:.2f}",
            save_top_k=2,
        )
    elif dataset == "nab":
        data_module = NABDataModule(
            dataset_name="ec2_disk_write_bytes_1ef3de",
            train_size=0.7,
            valid_size=0.12,
            test_size=0.1,
            seq_len=seq_len,
            batch_size=batch_size,
            step=step
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="valid_loss",
            dirpath="checkpoints/",
            filename="nab_anmaly-{epoch:02d}-{val_loss:.2f}",
            save_top_k=2,
        )
    else:
        data_module = YahooS5DataModule(dataset_path='datasets/datasets_files/yahoo_s5/A1Benchmark',
                                        dataset_name='A1Benchmark',
                                        dataset_configs_path='datasets/datasets_configs/yahoo_s5_config.json',
                                        train_size=0.6,
                                        valid_size=0.1,
                                        test_size=0.3,
                                        seq_len=seq_len,
                                        batch_size=batch_size,
                                        step=step,
                                        shuffle=False,
                                        normalize=True,
                                        normalize_range=(-1, 1))

        checkpoint_callback = ModelCheckpoint(
            monitor="valid_loss",
            dirpath="checkpoints/",
            filename="yahoo_s5-{epoch:02d}-{val_loss:.2f}",
            save_top_k=2,
        )

    early_stop_callback = EarlyStopping(
        monitor="valid_loss", min_delta=0.00, patience=3, verbose=True, mode="min"
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    call_backs = [early_stop_callback, checkpoint_callback, lr_monitor]

    trainer = Trainer(max_epochs=1, callbacks=call_backs, fast_dev_run=True)

    input_shape = (batch_size, seq_len, 1)
    model = LSTMAutoencoder(
        input_shape=input_shape,
        layers=[16, 8], 
        dropout=0.4,
        # anomaly_threshold=0.9, 
        anomaly_threshold="adaptive_threshold_detector"
    )

    data_module.prepare_data()
    data_module.setup()

    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )

    res = trainer.test(model, dataloaders=data_module.test_dataloader(), verbose=True)
    print(res)


if __name__ == "__main__":
    main()
