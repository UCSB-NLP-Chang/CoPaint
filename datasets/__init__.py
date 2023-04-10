from .celebahq import load_celebahq, load_lama_celebahq
from .imagenet import load_imagenet


REFERENCE_DIRS = {
    "celeba-hq": "datasets/lama-celeba/visual_test_source_256",
    "imagenet": "datasets/imagenet1kval/",
    "imagenet512": "datasets/imagenet1kval/",
}
