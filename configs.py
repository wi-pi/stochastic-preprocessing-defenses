from src.defenses import *

DEFENSES = {cls.__name__: cls for cls in InstancePreprocessorPyTorch.__subclasses__()}


def load_defense(defenses: list[str], k: int | None = None, params: list[str] | None = None):
    """Automatically load a defense.
    """
    # Eval params
    params_dict = {}
    for p in params or []:
        k, v = p.split('=', maxsplit=1)
        params_dict[k] = eval(v)

    """Single defense"""
    if len(defenses) == 1:
        return DEFENSES[defenses[0]](**params_dict)

    """Randomized ensemble of all"""
    defense = Ensemble(preprocessors=[DEFENSES[p]() for p in defenses], k=k)

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
