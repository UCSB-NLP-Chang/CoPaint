import torch
import lpips
import numpy as np


def check_device(tensor, device):
    if tensor.device != device:
        tensor = tensor.to(device)
    return tensor


def check_image(tensor):
    assert torch.max(tensor) <= 1.0 + 1e-3 and torch.min(tensor) >= -1.0 - 1e-3


class LPIPS:
    def __init__(self, base_model="alex", device="cpu") -> None:
        self.device = device
        self.loss_fn = lpips.LPIPS(net=base_model).to(device)

    @torch.no_grad()
    def score(self, samples: torch.Tensor, references: torch.Tensor):
        # ! Notice that samples and references should be in [-1, 1]
        check_image(samples)
        check_image(references)
        samples = check_device(samples, self.device)
        references = check_device(references, self.device)
        return self.loss_fn(samples, references).squeeze().detach().cpu()

    def on_dir():
        pass
