import abc
from typing import Callable, Optional

import numpy as np
import torch.nn as nn
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
from art.estimators.classification import PyTorchClassifier
from tqdm import trange

from src.defenses import EOT, RandomizedPreprocessor
from src.utils.functools import averaged_method


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

    def predict(self, x_test: np.ndarray, y_reference: np.ndarray, mode: str):
        # Initialize slots for prediction's correctness
        correct = []

        # Repeat prediction and logic OR the results
        # this means, the final prediction is correct as long as it corrects once.
        for _ in trange(self.repeat, desc=f'Predict ({mode})', leave=False):
            preds = self.model_defended.predict(x_test, batch_size=self.batch_size).argmax(1)
            correct.append(preds == y_reference)

        # Summarize repeated predictions
        if mode == 'all':
            correct = np.all(correct, axis=0)
        elif mode == 'any':
            correct = np.any(correct, axis=0)
        else:
            raise NotImplementedError(mode)

        return correct, np.mean(correct) * 100

    def attack(self, x_test: np.ndarray, y_reference: np.ndarray, adaptive: bool, mode: str, eot_samples: int = 1):
        # Do we attack defended or undefended model?
        defense = EOT(self.defense, nb_samples=eot_samples) if adaptive else None
        target_model = self.get_wrapper(self.model, defense=defense)

        # Adjust batch size according to EOT samples.
        batch_size = max(1, self.batch_size // eot_samples)

        # Final attack
        attack = self.attack_fn(target_model, batch_size=batch_size)
        x_adv = attack.generate(x_test, y_reference)

        return self.predict(x_adv, y_reference, mode)

    def attack_new(self, x_test: np.ndarray, y_reference: np.ndarray, adaptive: bool, mode: str, eot_samples: int = 1):
        # Do we attack defended or undefended model?
        defense = self.defense if adaptive else None
        target_model = self.get_wrapper(self.model, defense=defense)

        # Wrap up EOT
        target_model.loss_gradient = averaged_method(target_model.loss_gradient, n_calls=eot_samples)
        target_model.class_gradient = averaged_method(target_model.class_gradient, n_calls=eot_samples)

        # Final attack
        attack = self.attack_fn(target_model, batch_size=self.batch_size)
        x_adv = attack.generate(x_test, y_reference)

        return self.predict(x_adv, y_reference, mode)

    @staticmethod
    @abc.abstractmethod
    def get_wrapper(model: nn.Module, defense: Optional[PreprocessorPyTorch] = None) -> PyTorchClassifier:
        raise NotImplementedError
