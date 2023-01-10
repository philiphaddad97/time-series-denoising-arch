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
parser.add_argument("--batch_size", default="16", required=False)
parser.add_argument("--seq_len", default="24", required=False)
parser.add_argument("--step", default="24", required=False)
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
    
    else:
        data_module = YahooS5DataModule(dataset_path='datasets/datasets_files/yahoo_s5/A1Benchmark',
                                        dataset_name='A1Benchmark',
                                        dataset_configs_path='datasets/datasets_configs/yahoo_s5_config.json',
                                        train_size=0.1,
                                        valid_size=0.1,
                                        test_size=0.3,
                                        seq_len=seq_len,
                                        batch_size=batch_size,
                                        step=step,
                                        shuffle=False,
                                        normalize=True,
                                        normalize_range=(-1, 1))

    data_module.prepare_data()
    data_module.setup()
    import pdb; pdb.set_trace()
    

if __name__ == "__main__":
    main()
