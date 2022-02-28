from typing import List, Optional

from src.defenses import *

DEFENSES = {cls.__name__: cls for cls in InstancePreprocessorPyTorch.__subclasses__()}


def _ensemble(defenses: List[str], k: Optional[int] = None):
    defense = Ensemble(preprocessors=[DEFENSES[p]() for p in defenses], k=k)
    return defense


def load_defense(defenses: List[str], k: Optional[int] = None):
    """Single defense"""
    # defense = ResizePad(in_size=224, out_size=256)
    # defense = Crop(in_size=224, crop_size=128)
    # defense = DCT()
    # defense = Gaussian(kernel_size=(0, 6), sigma=(0.1, 1.1))

    """Randomized ensemble of all"""
    defense = _ensemble(defenses, k)

    """Manually specified ensemble"""
    # defense = Ensemble(
    #     preprocessors=[
    #         Gaussian(kernel_size=(0, 6), sigma=(1.0, 2.0)),
    #         Median(kernel_size=(0, 6)),
    #         JPEG(quality=(55, 75)),
    #     ],
    #     k=3,
    # )
    return defense
