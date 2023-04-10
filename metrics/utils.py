import torch


def check_image(tensor):
    assert (
        torch.max(tensor) <= 1.0 and torch.min(tensor) >= -1.0
    ), "Output images should be (-1, 1.)"


def normalize_tensor(tensor):
    check_image(tensor)
    return (tensor + 1.0) / 2.0
