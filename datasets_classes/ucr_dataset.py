from datasets_classes.abstract_classes import TimeSeriesDatasetSingleFile


class UCRDataset(TimeSeriesDatasetSingleFile):
    def __init__(self, data,
                 labels,
                 seq_len,
                 step,
                 mode):
        super().__init__(data=data,
                         labels=labels,
                         seq_len=seq_len,
                         step=step,
                         mode=mode)
