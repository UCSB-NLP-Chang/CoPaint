import torch
import numpy as np
import blobfile as bf


def normalize(image, shape=(256, 256)):
    """
    Given an PIL image, resize it and normalize each pixel into [-1, 1].
    Args:
        image: image to be normalized, PIL.Image
        shape: the desired shape of the image

    Returns: the normalized image

    """
    image = np.array(image.convert("RGB").resize(shape))
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = image * 2.0 - 1.0
    return image


# Copied from Repaint code
def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results
