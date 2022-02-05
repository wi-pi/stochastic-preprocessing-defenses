from typing import List

import art
import torch
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch


class PyTorchClassifier(art.estimators.classification.PyTorchClassifier):
    """
    This class fixes ART's PyTorchClassifier, which ignores `preprocess.estimate_forward`.
    """

    preprocessing_operations: List[PreprocessorPyTorch]

    def loss_gradient(self, x: torch.Tensor, y: torch.Tensor, training_mode: bool = False, **kwargs) -> torch.Tensor:
        # We only consider PyTorch preprocessors
        assert self.all_framework_preprocessing
        assert type(x) == type(y) == torch.Tensor

        # Set training mode
        self._model.train(mode=training_mode)

        # Detach input tensors
        x = x.clone().detach()
        y = y.clone().detach()

        # Forward pass: preprocessing + model
        x_processed, y_processed = self._apply_preprocessing(x, y, fit=False, no_grad=True)
        x_processed.requires_grad = True
        y_processed = self.reduce_labels(y_processed)
        model_outputs = self._model(x_processed)

        # Backward pass: model
        loss = self._loss(model_outputs[-1], y_processed)
        self._model.zero_grad()
        grads, = torch.autograd.grad(loss, x_processed)

        # Forward pass: preprocessing (BPDA)
        x.requires_grad = True
        x_processed = self._apply_preprocessing_estimate(x)

        # Backward pass: preprocessing (BPDA)
        x_processed.backward(grads)
        grads = x.grad

        # Sanity check
        assert grads is not None
        assert grads.shape == x.shape

        return grads

    def _apply_preprocessing_estimate(self, x: torch.Tensor) -> torch.Tensor:
        for preprocess in self.preprocessing_operations:
            x = preprocess.estimate_forward(x)
        return x
