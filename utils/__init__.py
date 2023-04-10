import torch
from PIL import Image
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage


def normalize_image(tensor_img):
    tensor_img = (tensor_img + 1.0) / 2.0
    return tensor_img


def save_grid(tensor_img, path, nrow=5):
    """
    tensor_img: [B, 3, H, W] or [tensor(3, H, W)]
    """
    if isinstance(tensor_img, list):
        tensor_img = torch.stack(tensor_img)
    assert len(tensor_img.shape) == 4
    tensor_img = tensor_img.clamp(min=0.0, max=1.0)
    grid = make_grid(tensor_img, nrow=nrow)
    pil = ToPILImage()(grid)
    pil.save(path)


def save_image(tensor_img, path):
    """
    tensor_img : [3, H, W]
    """
    tensor_img = tensor_img.clamp(min=0.0, max=1.0)
    pil = ToPILImage()(tensor_img)
    pil.save(path)
