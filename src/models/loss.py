import torch
from torch.nn.modules.loss import _Loss


class LinearLoss(_Loss):
    """
    Experimental implementation of Linear Loss.

    TODO: Rigorously test this loss function.

    | Paper link: https://openreview.net/pdf?id=p4SrFydwO5
    """

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # utility index at the batch dim
        batch_ind = torch.arange(y_pred.shape[0])

        # get true class's index and logit
        y_true_ind = torch.argmax(y_true, dim=1)
        y_true_val = y_pred[batch_ind, y_true_ind]

        # get top-2 prediction's index and logit
        y_pred_top2_val, y_pred_top2_ind = torch.topk(y_pred, k=2, dim=1)

        # take the first (i.e., largest) logit whose ind is not the true class's ind.
        ind = torch.not_equal(y_pred_top2_ind, y_true_ind[..., None]).float().argmax(dim=1)
        y_false_max_val = y_pred_top2_val[batch_ind, ind]

        return torch.mean(y_false_max_val - y_true_val)
