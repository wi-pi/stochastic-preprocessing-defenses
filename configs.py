from src.defenses import *

DEFENSES = {cls.__name__: cls for cls in InstancePreprocessorPyTorch.__subclasses__()}


def load_defense(defenses: list[str], nb_samples: int | None = None, params: list[str] | None = None):
    """
    Automatically load a defense.

    :param defenses: List of preprocessing defenses.
    :param nb_samples: Number of defenses sampled in each inference.
    :param params: Additional kwargs passed to a single defense.
    :return:
    """

    # Parse params
    def _split(expr: str):
        k, v = expr.split('=', maxsplit=1)
        return k, eval(v)

    match len(defenses):

        # Hard-coded ensemble
        case 0:
            defense = Ensemble(
                preprocessors=[
                    Gaussian(kernel_size=(0, 6), sigma=(1.0, 2.0)),
                    Median(kernel_size=(0, 6)),
                    JPEG(quality=(55, 75)),
                ],
                nb_samples=nb_samples,
            )

        # Single defense with additional kwargs
        case 1:
            kwargs = dict(map(_split, params or []))
            defense = DEFENSES[defenses[0]](**kwargs)

        # Randomized ensemble of all
        case _:
            defense = Ensemble(preprocessors=[DEFENSES[p]() for p in defenses], nb_samples=nb_samples)

    return defense
