import torch
from torchmetrics import F1Score


class AdjustedF1Score(F1Score):
    def __init__(self):
        super().__init__()

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # replace each row with 1 if any of the values is 1 and 0 otherwise
        row_sum_preds = torch.sum(preds, dim=1)
        row_sum_target = torch.sum(target, dim=1)

        accumulated_preds = torch.where(row_sum_preds > 0, torch.ones_like(row_sum_preds), torch.zeros_like(row_sum_preds))
        accumulated_target = torch.where(row_sum_target > 0, torch.ones_like(row_sum_target), torch.zeros_like(row_sum_target))

        accumulated_preds = accumulated_preds.int()
        accumulated_target = accumulated_target.int()

        super().update(accumulated_preds, accumulated_target)
