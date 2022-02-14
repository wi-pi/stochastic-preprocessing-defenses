import abc
from typing import Callable, Optional

import numpy as np
import torch.nn as nn
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
from tqdm import trange

from src.defenses import EOT, RandomizedPreprocessor


class BaseTestKit(abc.ABC):

    def __init__(
        self,
        model: nn.Module,
        defense: Optional[RandomizedPreprocessor],
        attack_fn: Callable,
        batch_size: int,
        repeat: int,
    ):
        self.model = model
        self.defense = defense
        self.model_defended = self.get_wrapper(model, defense=defense)
        self.attack_fn = attack_fn
        self.batch_size = batch_size
        self.repeat = repeat

    def predict(self, x_test: np.ndarray, y_test: np.ndarray):
        # Initialize slots for prediction's correctness
        correct = np.zeros_like(y_test)

        # Repeat prediction and logic OR the results
        # this means, the final prediction is correct as long as it corrects once.
        for _ in trange(self.repeat, desc='Union Predict', leave=False):
            preds = self.model_defended.predict(x_test, batch_size=self.batch_size).argmax(1)
            correct |= preds == y_test

        return correct, np.mean(correct) * 100

    def attack(self, x_test: np.ndarray, y_test: np.ndarray, adaptive: bool, eot_samples: int = 1):
        # Do we attack defended or undefended model?
        defense = EOT(self.defense, nb_samples=eot_samples) if adaptive else None
        target_model = self.get_wrapper(self.model, defense=defense)

        # Adjust batch size according to EOT samples.
        batch_size = max(1, self.batch_size // eot_samples)

        # Final attack
        attack = self.attack_fn(target_model, batch_size=batch_size)
        x_adv = attack.generate(x_test, y_test)

        return self.predict(x_adv, y_test)

    @staticmethod
    @abc.abstractmethod
    def get_wrapper(model: nn.Module, defense: Optional[PreprocessorPyTorch] = None):
        raise NotImplementedError
