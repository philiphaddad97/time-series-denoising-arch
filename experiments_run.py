from typing import Dict, List
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)

from data_modules import YahooS5DataModule, NABDataModule, UCRDataModule
from models import LSTMAutoencoder
from utils import load_json

# TODO: implement it as args for CLI
EXP_IDS = [1]  # To run all the exp keep it empty
SUFFIX = "nab_2"
LOG_DIR = f"logs_{SUFFIX}"
CHECKPOINTS_DIR = f"checkpoints_{SUFFIX}"
EXP_CONFIG_FILE_PATH = "./nab_experiments_config.json"
NUM_WORKERS = 8
FAST_DEV_RUN = False
ACCELERATOR = "gpu"


def filter_exp(configs: List[Dict], exp_ids: List[int] = []) -> List[Dict]:
    """
    Filters a list of experiment configurations based on the provided experiment IDs.
    Parameters:
    - configs (List[Dict]): A list of dictionaries representing experiment configurations.
    - exp_ids (List[intttra], optional): A list of experiment IDs to filter by. If not provided, returns all experiment configurations.

    Returns:
    - List[Dict]: A list of experiment configurations that match the provided experiment IDs.
    """
    if len(exp_ids) == 0:
        return configs
    return list(filter(lambda d: d["id"] in exp_ids, configs))


def run_exp(exp: Dict) -> None:
    """
    Runs an experiment based on the provided configuration.

    Parameters:
    - exp (Dict): A dictionary containing the configuration for the experiment. The dictionary should include the following keys:
        - dataset': A dictionary containing the name and other relevant information for the dataset to be used in the experiment.
        - 'seq_len': The length of the sequences to be used in the experiment.
        - 'batch_size': The batch size to be used in the experiment.
        - 'step_size': The step size to be used in the experiment.
        - 'patience': The number of epochs to wait before early stopping.
        - 'epochs': The maximum number of epochs to run the experiment for.
        - 'layers': The number of layers to use in the model.
        - 'dropout': The dropout rate to use in the model.
        - 'threshold': The anomaly threshold to use in the model.
        - 'optimizer': The optimizer to use in the model.

    Returns: None
    """
    if exp["dataset"]["name"] == "ucr":
        data_module = UCRDataModule(
            dataset_file_path=exp["dataset"]["path"],
            valid_size=exp["dataset"]["split"][1],
            seq_len=exp["seq_len"],
            batch_size=exp["batch_size"],
            step=exp["step_size"],
            num_workers=NUM_WORKERS,
        )
    elif exp["dataset"]["name"] == "nab":
        data_module = NABDataModule(
            dataset_name=exp["dataset"]["sub_name"],
            train_size=exp["dataset"]["split"][0],
            valid_size=exp["dataset"]["split"][1],
            test_size=exp["dataset"]["split"][2],
            seq_len=exp["seq_len"],
            batch_size=exp["batch_size"],
            step=exp["step_size"],
            num_workers=NUM_WORKERS,
        )
    else:
        data_module = YahooS5DataModule(
            dataset_path=exp["dataset"]["path"],
            dataset_name=exp["dataset"]["sub_name"],
            dataset_configs_path="datasets/datasets_configs/yahoo_s5_config.json",
            train_size=exp["dataset"]["split"][0],
            valid_size=exp["dataset"]["split"][1],
            test_size=exp["dataset"]["split"][2],
            seq_len=exp["seq_len"],
            batch_size=exp["batch_size"],
            step=exp["step_size"],
            shuffle=False,
            normalize=True,
            normalize_range=(-1, 1),
            num_workers=NUM_WORKERS,
        )

    data_module.prepare_data()
    data_module.setup()

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath=f"{CHECKPOINTS_DIR}/exp_{exp['id']}/",
        save_top_k=2,
    )

    early_stop_callback = EarlyStopping(
        monitor="valid_loss",
        min_delta=0.00,
        patience=exp["patience"],
        verbose=True,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    call_backs = [early_stop_callback, checkpoint_callback, lr_monitor]

    logger = TensorBoardLogger(LOG_DIR, name=f"model_exp_{exp['id']}")
    trainer = Trainer(
        max_epochs=exp["epochs"],
        callbacks=call_backs,
        fast_dev_run=FAST_DEV_RUN,
        logger=logger,
        accelerator=ACCELERATOR,
    )

    input_shape = (exp["batch_size"], exp["seq_len"], 1)
    model = LSTMAutoencoder(
        input_shape=input_shape,
        layers=exp["layers"],
        dropout=exp["dropout"],
        anomaly_threshold=exp["threshold"],
        optimizer=exp["optimizer"],
    )

    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )

    res = trainer.test(model, dataloaders=data_module.test_dataloader(), verbose=True)

    logger.log_hyperparams(exp)
    logger.log_metrics(res[0])
    logger.finalize("success")


def main():
    exp_list = load_json(EXP_CONFIG_FILE_PATH)
    exp_list = filter_exp(exp_list, EXP_IDS)
    for exp in exp_list:
        # TODO: Implement Multi-Processing
        run_exp(exp)


if __name__ == "__main__":
    main()
