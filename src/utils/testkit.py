import abc
from typing import Callable

import numpy as np
import scipy.stats
import torch.nn as nn
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
from art.estimators.classification import PyTorchClassifier
from loguru import logger
from tqdm import trange

from src.utils.functools import averaged_method


class BaseTestKit(abc.ABC):
    """
    Base class for the test kit.
    """

    supported_modes = ['any', 'all', 'vote']

    def __init__(
        self,
        model: nn.Module,
        defense: PreprocessorPyTorch | None,
        attack_fn: Callable,
        *,
        batch_size: int,
        mode: str,
        nb_repeats: int,
    ):
        """
        Base class for test kit.
        
        :param model: Tested PyTorch model.
        :param defense: Optional preprocessing defense.
        :param attack_fn: Function that takes an ART estimator to initialize the attacker.
        :param batch_size: Test batch size.
        :param mode: Aggregation mode, support "any", "all", or "vote".
        :param nb_repeats: Number of repeats for each prediction.
        """
        self.model = model
        self.defense = defense
        self.estimator = self.get_estimator(model, defense=defense)
        self.attack_fn = attack_fn
        self.batch_size = batch_size
        self.mode = mode
        self.nb_repeats = nb_repeats

        if mode not in self.supported_modes:
            raise NotImplementedError(f'Prediction mode not supported: {mode}.')

    def predict(self, x_test: np.ndarray, y_test: np.ndarray, *, error_rate: bool = False) -> tuple[np.ndarray, float]:
        """
        Predict test samples with reference labels.

        :param x_test: Test samples of shape (B, C, H, W).
        :param y_test: Reference labels of shape (B,), could be ground truth labels or target labels.
        :param error_rate: Return "error rate" instead of "accuracy".
        :return: Indicator and percentage of successful predictions.
        """
        # Initialize slots for repeated predictions
        y_pred_all = []

        # Repeat prediction for multiple times
        for _ in trange(self.nb_repeats, desc=f'Predict ({self.mode})', leave=False):
            y_pred = self.estimator.predict(x_test, batch_size=self.batch_size).argmax(1)
            y_pred_all.append(y_pred)

        y_pred_all = np.stack(y_pred_all)

        # Summarize repeated predictions
        match self.mode:
            case 'all':
                match = np.all(y_pred_all == y_test[None], axis=0)

            case 'any':
                match = np.any(y_pred_all == y_test[None], axis=0)

            case 'vote':
                preds_vote, _ = scipy.stats.mode(y_pred_all, axis=0)
                match = preds_vote[0] == y_test

            case _:
                raise RuntimeError(f'Prediction mode not supported: {self.mode}.')

        # Define the notion of success
        success = ~match if error_rate else match
        return success, np.mean(success) * 100

    def attack(self, x_test: np.ndarray, y_test: np.ndarray, *, targeted: bool, adaptive: bool, eot: int = 1):
        """
        Attack test samples with reference labels.

        :param x_test: Test samples of shape (B, C, H, W).
        :param y_test: Reference labels of shape (B,), could be ground truth labels or target labels.
        :param targeted: If True, this is a targeted attack.
        :param adaptive: If True, attack the defense jointly.
        :param eot: Number of EOT samples when computing the gradients.
        :return: Indicator and percentage of matching with the reference label.
        """
        # Do we attack defended or undefended model?
        defense = self.defense if adaptive else None
        estimator = self.get_estimator(self.model, defense=defense)

        # Wrap gradient methods with EOT
        self.prepare_estimator_for_eot(estimator, eot)

        # Final attack
        attack = self.attack_fn(estimator, batch_size=self.batch_size)
        if id(estimator) != id(attack.estimator):
            logger.warning(f'The attack changed your estimator, make sure you prepared the correct instance for EOT.')
        x_adv = attack.generate(x_test, y_test)

        return self.predict(x_adv, y_test, error_rate=not targeted)

    def test(self, x_test: np.ndarray, y_test: np.ndarray, *, targeted: bool, test_non_adaptive: bool, eot: int):
        """
        Test attacks.

        :param x_test: Test samples of shape (B, C, H, W).
        :param y_test: Ground truth labels of shape (B,).
        :param targeted: If True, this is a targeted attack.
        :param test_non_adaptive: If True, test non-adaptive attacks.
        :param eot: Number of EOT samples when computing the gradients.
        :return: None
        """
        logger.debug(f'Test attack with {targeted = } and {eot = }.')

        # Test benign
        success, success_rate = self.predict(x_test, y_test, error_rate=not targeted)
        logger.info(f'Attack Success Rate (benign): {success_rate:.2f}')

        # Select samples that we need to attack
        indices = np.nonzero(~success)
        x_test = x_test[indices]
        y_test = y_test[indices]

        # Test adversarial (non-adaptive)
        if test_non_adaptive:
            success, success_rate = self.attack(x_test, y_test, targeted=targeted, adaptive=False, eot=1)
            logger.info(f'Attack Success Rate (non-adaptive): {success_rate:.2f}')

        # Test adversarial (adaptive)
        success, success_rate = self.attack(x_test, y_test, targeted=targeted, adaptive=True, eot=eot)
        logger.info(f'Attack Success Rate (adaptive): {success_rate:.2f}')

    @staticmethod
    @abc.abstractmethod
    def get_estimator(model: nn.Module, defense: PreprocessorPyTorch | None = None) -> PyTorchClassifier:
        """
        Produce the estimator with the given defense.

        :param model: PyTorch model.
        :param defense: Optional preprocessing defense.
        :return: Initialized ART estimator.
        """
        raise NotImplementedError

    @staticmethod
    def prepare_estimator_for_eot(estimator: PyTorchClassifier, eot_samples: int = 1):
        """
        Modify an estimator so that it is ready for EOT attack.

        By default, the gradient methods are repeated and averaged.

        :param estimator: PyTorch classifier to be attacked.
        :param eot_samples: Number of EOT samples when computing the gradients.
        :return: None
        """
        estimator.loss_gradient = averaged_method(estimator.loss_gradient, n_calls=eot_samples)
        estimator.class_gradient = averaged_method(estimator.class_gradient, n_calls=eot_samples)
