import cv2
import torch
from functools import partial
import numpy as np


def generate_half_mask(shape):
    assert len(shape) == 2
    assert shape[1] % 2 == 0
    half = shape[1] // 2
    ret = [[(0 if c >= half else 1) for c in range(shape[1])]
           for r in range(shape[0])]
    ret = torch.tensor(ret, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return ret


def generate_alternate_mask(shape):
    assert len(shape) == 2
    assert shape[1] % 2 == 0
    ret = [[1 if r % 2 == 1 else 0 for c in range(
        shape[1])] for r in range(shape[0])]
    ret = torch.tensor(ret, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return ret


def generate_sp2_mask(shape):
    assert len(shape) == 2
    assert shape[1] % 2 == 0
    ret = [
        [1 if c % 2 == 1 and r % 2 == 1 else 0 for c in range(shape[1])]
        for r in range(shape[0])
    ]
    ret = torch.tensor(ret, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return ret


def generate_center_mask(shape):
    assert len(shape) == 2
    assert shape[1] % 2 == 0
    center = shape[0] // 2
    center_size = shape[0] // 4
    half_resol = center_size // 2  # for now
    ret = torch.zeros(shape, dtype=torch.float32)
    ret[
        center - half_resol: center + half_resol,
        center - half_resol: center + half_resol,
    ] = 1
    ret = ret.unsqueeze(0).unsqueeze(0)
    return ret


def generate_random_mask(shape):
    # TODO: provide torch.generator to fix the mask
    assert len(shape) == 2
    assert shape[1] % 2 == 0
    ret = torch.randint(0, 2, shape)
    ret = ret.unsqueeze(0).unsqueeze(0)
    return ret


def generate_full_mask(shape):
    # ! debug usage ONLY, check if DDIM sampling and DPM sampling code is right
    # ! full mask corresponds to unconditional generation
    assert len(shape) == 2
    assert shape[1] % 2 == 0
    ret = torch.zeros(shape)
    ret = ret.unsqueeze(0).unsqueeze(0)
    return ret


def generate_text_mask(shape, text_type):
    if text_type == "lorem":
        mask_path = "datasets/text_masks/lorem3.npy"
    elif text_type == "cat":
        mask_path = "datasets/text_masks/lolcat_extra.npy"
    mask = np.load(mask_path)
    if mask.shape != shape:
        mask = cv2.resize(mask, dsize=shape, interpolation=cv2.INTER_CUBIC)
    assert mask.shape == shape
    mask = torch.from_numpy(mask).to(torch.float32)
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask


def generate_thin_mask(shape, result_path):
    # ! this should  be loadded in load_lama_celebahq function
    pass


def generate_thick_mask(shape, result_path):
    # ! this should  be loadded in load_lama_celebahq function
    pass


mask_generators = {
    "half": generate_half_mask,
    "line": generate_alternate_mask,
    "sr2": generate_sp2_mask,
    "expand": generate_center_mask,
    "text": partial(generate_text_mask, text_type="lorem"),
    "random": generate_random_mask,
    "text_cat": partial(generate_text_mask, text_type="cat"),
    "full": generate_full_mask,
}
