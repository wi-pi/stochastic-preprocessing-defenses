import torch

from src.defenses import DEFENSES, Ensemble


def get_defense(defenses):
    ensemble = Ensemble(
        randomized=True,
        preprocessors=[DEFENSES[p].as_randomized() for p in defenses],
        k=len(defenses),
    )

    def wrapper(x: torch.Tensor):
        x, _ = ensemble.forward(x[None], None)
        return x[0]

    return wrapper
