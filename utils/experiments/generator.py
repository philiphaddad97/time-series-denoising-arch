import json
from itertools import product
from typing import Dict, List, Union

# TODO: implement it as args for CLI
DATASET = "ucr"


def generate_dataset_choices(dataset: str) -> List[Dict[str, Union[str, List[float]]]]:
    """
    Generates a list of dataset configurations for the specified dataset.

    Parameters:
    - dataset (str): The name of the dataset to generate configurations for. Can be one of 'ucr', 'nab', or 's5'.

    Returns:
    - List[Dict[str, Union[str, List[float]]]]: A list of dictionaries representing the generated dataset configurations.
        Each dictionary has the following keys:
            - 'name (str)': The name of the dataset.
            - 'sub_name (str)': The sub-name of the dataset, if applicable.
            - 'path (str)': The path to the dataset file(s).
            - 'split (List[int])': A list of three floats representing the train, validation, and test split ratios, respectively.
    """
    name = [dataset]
    split = [[0.6, 0.1, 0.3]]
    if dataset == "nab":
        sub_name = [
            "ec2_cpu_utilization_24ae8d",
        ]
        path = ["N/A"]  # Not used with ucr dataset ("any value is acceptable")
    elif dataset == "ucr":
        sub_name = ["N/A"]  # Not used with ucr dataset ("any value is acceptable")
        path = [
            "datasets/datasets_files/ucr_anomaly/145_UCR_Anomaly_Lab2Cmac011215EPG1_5000_17210_17260.txt",
        ]

    else:
        sub_name = ["A4Benchmark"]
        path = [
            # "datasets/datasets_files/yahoo_s5/A1Benchmark",
            # "datasets/datasets_files/yahoo_s5/A3Benchmark",
            # "datasets/datasets_files/yahoo_s5/A4Benchmark",
        ]

    choices = []
    for name_ch, sub_name_ch, path_ch, split_ch in product(name, sub_name, path, split):
        choices.append(
            {
                "name": name_ch,
                "sub_name": sub_name_ch,
                "path": path_ch,
                "split": split_ch,
            }
        )

    return choices


def generate(dataset: str) -> List[Dict]:
    """
    Generates a list of experiment configurations for the specified dataset.

    Parameters:
    - dataset (str): The name of the dataset to generate configurations for. Can be one of 'ucr', 'nab', or 'yahoo_s5'.

    Returns:
    - List[Dict]: A list of dictionaries representing the generated experiment configurations. Each dictionary has the following keys:
        - 'id': A unique identifier for the experiment.
        - 'dataset': A dictionary containing the name and other relevant information for the dataset to be used in the experiment.
        - 'seq_len': The length of the sequences to be used in the experiment.
        - 'step_size': The step size to be used in the experiment.
        - 'layers': The number of layers to use in the model.
        - 'dropout': The dropout rate to use in the model.
        - 'optimizer': The optimizer to use in the model.
        - 'threshold': The anomaly threshold to use in the model.
        - 'epochs': The maximum number of epochs to run the experiment for.
        - 'patience': The number of epochs to wait before early stopping.
        - 'batch_size': The batch size to be used in the experiment.
    """
    # Dataset choices
    dataset_choices = generate_dataset_choices(dataset)
    seq_len = [24]
    step_size = [1]

    # Model choices
    layers_choices = [[16, 8]]
    dropout_choices = [0.2]
    optimizer_choices = ["Adam"]
    threshold_choices = [0.99, 0.97,"adaptive_threshold_detector"]

    # Training choices
    epochs_choices = [15]
    patience_choices = [5]
    batch_size_choices = [32]

    all_choices = product(
        dataset_choices,
        seq_len,
        step_size,
        layers_choices,
        dropout_choices,
        optimizer_choices,
        threshold_choices,
        epochs_choices,
        patience_choices,
        batch_size_choices,
    )

    choices = []
    idx = 0
    for (
        dataset_ch,
        seq_len_ch,
        step_size_ch,
        layers_ch,
        dropout_ch,
        optimizer_ch,
        threshold_ch,
        epochs_ch,
        patience_ch,
        batch_size_ch,
    ) in all_choices:
        idx += 1
        choices.append(
            {
                "id": idx,
                "dataset": dataset_ch,
                "seq_len": seq_len_ch,
                "step_size": step_size_ch,
                "layers": layers_ch,
                "dropout": dropout_ch,
                "optimizer": optimizer_ch,
                "threshold": threshold_ch,
                "epochs": epochs_ch,
                "patience": patience_ch,
                "batch_size": batch_size_ch,
            }
        )

    return choices


if __name__ == "__main__":
    generated_exp = generate(DATASET)
    with open(f"{DATASET}_experiments_config.json", "w") as f:
        json.dump(generated_exp, f)
