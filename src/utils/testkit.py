import abc
from typing import Callable, Optional

import numpy as np
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

    def __init__(
        self,
        model: nn.Module,
        defense: Optional[PreprocessorPyTorch],
        attack_fn: Callable,
        batch_size: int,
        nb_repeats: int,
    ):
        """
        Base class for test kit.
        
        :param model: Tested PyTorch model.
        :param defense: Optional preprocessing defense.
        :param attack_fn: Function that takes an ART estimator to initialize the attacker.
        :param batch_size: Test batch size.
        :param nb_repeats: Number of repeats for each prediction.
        """
        self.model = model
        self.defense = defense
        self.estimator = self.get_estimator(model, defense=defense)
        self.attack_fn = attack_fn
        self.batch_size = batch_size
        self.nb_repeats = nb_repeats

    def predict(self, x_test: np.ndarray, y_reference: np.ndarray, mode: str):
        """
        Predict test samples with reference labels.

        :param x_test: Test samples of shape (B, C, H, W).
        :param y_reference: Reference labels of shape (B,), could be ground truth labels or target labels.
        :param mode: Aggregation mode, either "any" or "all".
        :return: Indicator and percentage of matching with the reference label.
        """
        # Initialize slots for prediction's correctness
        correct = []

        # Repeat prediction and logic OR the results
        # this means, the final prediction is correct as long as it corrects once.
        for _ in trange(self.nb_repeats, desc=f'Predict ({mode})', leave=False):
            preds = self.estimator.predict(x_test, batch_size=self.batch_size).argmax(1)
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
        """
        Attack test samples with reference labels.

        :param x_test: Test samples of shape (B, C, H, W).
        :param y_reference: Reference labels of shape (B,), could be ground truth labels or target labels.
        :param adaptive: If True, attack the defense jointly.
        :param mode: Aggregation mode, either "any" or "all".
        :param eot_samples: Number of EOT samples when computing the gradients.
        :return: Indicator and percentage of matching with the reference label.
        """
        # Do we attack defended or undefended model?
        defense = self.defense if adaptive else None
        estimator = self.get_estimator(self.model, defense=defense)

        # Wrap gradient methods with EOT
        self.prepare_estimator_for_eot(estimator, eot_samples)

        # Final attack
        attack = self.attack_fn(estimator, batch_size=self.batch_size)
        x_adv = attack.generate(x_test, y_reference)

        return self.predict(x_adv, y_reference, mode)

    def test_untargeted(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray,
        test_non_adaptive: bool,
        eot_samples: int,
        mode: str = 'all',
    ):
        """
        Test untargeted attacks.

        :param x_test: Test samples of shape (B, C, H, W).
        :param y_test: Ground truth labels of shape (B,).
        :param test_non_adaptive: If True, test non-adaptive attacks.
        :param eot_samples: Number of EOT samples when computing the gradients.
        :param mode: Aggregation mode, the attack succeeds in "any" or "all" of the multiple predictions.
        :return: None
        """
        logger.debug('Test with no targets.')

        # 1. Test benign
        preds_clean, acc = self.predict(x_test, y_test, mode=mode)
        logger.info(f'Accuracy: {acc:.2f}')

        # 2. Select correctly classified samples
        indices = np.nonzero(preds_clean)
        x_test = x_test[indices]
        y_test = y_test[indices]

        # 3. Test adversarial (non-adaptive)
        if test_non_adaptive:
            preds_adv, rob = self.attack(x_test, y_test, adaptive=False, mode=mode)
            logger.info(f'Robustness (non-adaptive): {rob:.2f}')

        # 4. Test adversarial (adaptive)
        preds_adv, rob = self.attack(x_test, y_test, adaptive=True, mode=mode, eot_samples=eot_samples)
        logger.info(f'Robustness (adaptive): {rob:.2f}')

    def test_targeted(
        self,
        x_test: np.ndarray,
        y_target: np.ndarray,
        test_non_adaptive: bool,
        eot_samples: int,
        mode: str = 'all',
    ):
        """
        Test targeted attacks.

        :param x_test: Test samples of shape (B, C, H, W).
        :param y_target: Target labels of shape (B,).
        :param test_non_adaptive: If True, test non-adaptive attacks.
        :param eot_samples: Number of EOT samples when computing the gradients.
        :param mode: Aggregation mode, the attack succeeds in "any" or "all" of the multiple predictions.
        :return: None
        """
        logger.debug(f'Test with target.')

        # 1. Test benign
        preds_clean, asr = self.predict(x_test, y_target, mode=mode)
        logger.info(f'Attack Success Rate (benign): {asr:.2f}')

        # 2. Select unsuccessful attack samples
        indices = np.nonzero(preds_clean == 0)
        x_test = x_test[indices]
        y_target = y_target[indices]

        # 3. Test adversarial (non-adaptive)
        if test_non_adaptive:
            preds_adv, asr = self.attack(x_test, y_target, adaptive=False, mode=mode)
            logger.info(f'Attack Success Rate (non-adaptive): {asr:.2f}')

        # 4. Test adversarial (adaptive)
        preds_adv, asr = self.attack(x_test, y_target, adaptive=True, mode=mode, eot_samples=eot_samples)
        logger.info(f'Attack Success Rate (adaptive): {asr:.2f}')

    @staticmethod
    @abc.abstractmethod
    def get_estimator(model: nn.Module, defense: Optional[PreprocessorPyTorch] = None) -> PyTorchClassifier:
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
