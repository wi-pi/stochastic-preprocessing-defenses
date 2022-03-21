from src.defenses import *

DEFENSES = {cls.__name__: cls for cls in InstancePreprocessorPyTorch.__subclasses__()}


def _split_param(expr: str):
    k, v = expr.split('=', maxsplit=1)
    return k, eval(v)


def load_defense(defenses: list[str], nb_samples: int | None = None, params: list[str] | None = None):
    """Automatically load a defense.
    """
    # Parse params
    params_dict = dict(map(_split_param, params))

    """Single defense"""
    if len(defenses) == 1:
        return DEFENSES[defenses[0]](**params_dict)

    """Randomized ensemble of all"""
    defense = Ensemble(preprocessors=[DEFENSES[p]() for p in defenses], nb_samples=nb_samples)

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
