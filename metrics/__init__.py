import torch
from .lpips import LPIPS
from .psnr import PSNR
from .ssim import SSIM
from typing import List


def reduce_results(dataset_scores: List[torch.Tensor], eval_type):
    # batch_results: a list of batch scores for a certain metric
    # return: over all mean, colmin_mean
    dataset_scores = torch.stack(dataset_scores)
    mean = torch.mean(dataset_scores).item()
    if eval_type == "min":
        colmin_mean_scores = torch.mean(dataset_scores.min(dim=-1)[0]).item()
    elif eval_type == "max":
        colmin_mean_scores = torch.mean(dataset_scores.max(dim=-1)[0]).item()

    return mean, colmin_mean_scores


# avoid shape of the score is zero
def score_reshape(s): return s.unsqueeze() if s.dim() == 0 else s


class Metric:
    def __init__(self, metric_fn, eval_type="min") -> None:
        self.metric_name = metric_fn.__class__
        self.metric_fn = metric_fn
        self.dataset_scores = []
        self.eval_type = eval_type

    def update(self, samples, references):
        batch_res = self.metric_fn.score(samples, references)
        if len(batch_res.shape) == 0:
            batch_res = torch.tensor([batch_res])
        self.dataset_scores.append(batch_res)

    def report_batch(self):
        # report current batch result
        batch_res = " ".join(["%.3lf" % i for i in self.dataset_scores[-1]])
        return batch_res

    def report_all(self):
        mean, colmin_mean = reduce_results(self.dataset_scores, self.eval_type)
        return mean, colmin_mean

    def reset(self):
        self.dataset_scores = []
