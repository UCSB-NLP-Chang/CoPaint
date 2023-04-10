import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image

from datasets.masks import mask_generators
from datasets.utils import _list_image_files_recursively


# copied from Repaint code
def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def normalize_arr(arr_image):
    arr_image = arr_image.astype(np.float32) / 255.0
    arr_image = arr_image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(arr_image)
    image = image * 2.0 - 1.0
    return image


def load_imagenet(
    offset=0, max_len=100, shape=(256, 256), mask_type="half", split="test"
):
    gt_dir = os.path.join(os.getcwd(), f"./datasets/imagenet100/{split}/")
    gt_paths = _list_image_files_recursively(gt_dir)
    gt_paths.sort()

    def load_image(path): return normalize_arr(
        center_crop_arr(Image.open(path).convert("RGB"), shape[0])
    )
    rawlabels = json.load(open(os.path.join(gt_dir, f"../val_label.json")))
    labels = {}
    for k, v in rawlabels.items():
        k = k.replace("_val_", "_test_")
        labels[k] = v
    if mask_type not in ["narrow", "wide"]:
        # simple masks
        mask_generator = mask_generators[mask_type]

        def load_image_mask_name(path): return (
            load_image(path),
            mask_generator(shape),
            "%05d" % int(os.path.splitext(
                os.path.basename(path))[0].split("_")[-1]),
            labels[os.path.basename(path)],
        )
        res = [
            load_image_mask_name(path) for path in gt_paths[offset: offset + max_len]
        ]
    else:
        # thin/thick masks
        mask_dir = os.path.join(
            os.getcwd(), f"datasets/Repaint_mask/{mask_type}")
        mask_paths = _list_image_files_recursively(mask_dir)
        mask_paths.sort()

        def load_mask(path): return torch.from_numpy(
            np.array(Image.open(path).resize(
                shape).convert("L"), dtype=np.float32)
            / 255.0,
        )

        def load_image_mask_name(i): return (
            load_image(gt_paths[i]),
            load_mask(mask_paths[i]),
            "%05d"
            % int(os.path.splitext(os.path.basename(gt_paths[i]))[0].split("_")[-1]),
            labels[os.path.basename(gt_paths[i])],
        )
        res = [load_image_mask_name(i) for i in range(offset, offset + max_len)]

    return res
