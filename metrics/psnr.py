import torch
from .utils import normalize_tensor


class PSNR:
    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def score(self, samples: torch.Tensor, references: torch.Tensor):
        # samples: B, C, H, W
        # references: 1, C, H, W or B, C, H, W
        B = samples.shape[0]
        samples = normalize_tensor(samples)
        references = normalize_tensor(references)
        if references.shape[0] == 1:
            references = references.repeat(B, 1, 1, 1)

        mse = torch.mean((samples - references) ** 2, dim=(1, 2, 3))
        peak = 1.0  # we normalize the image to (0., 1.)
        psnr = 10 * torch.log10(peak / mse)
        return psnr.detach().cpu()
