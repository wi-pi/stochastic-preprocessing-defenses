import types
from typing import Optional

import torch
from art.estimators.classification import PyTorchClassifier


def loss_gradient_average_logits(instance: PyTorchClassifier, nb_samples: int):
    """
    Return a loss_gradient method with loss computed on "averaged logits".

    :param instance: Instance whose loss_gradient to be modified.
    :param nb_samples: Number of samples to compute the average logits.
    :return: New bounded loss_gradient method of the given instance.
    """

    assert isinstance(instance, PyTorchClassifier)

    def loss_gradient(self: PyTorchClassifier, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:

        assert isinstance(self, PyTorchClassifier)
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

        self._model.train(mode=False)

        # Leaf node
        x_grad = x.clone().detach().requires_grad_(True)
        y_grad = y.clone().detach()

        def get_logits(_y: Optional[torch.Tensor] = None):
            inputs_t, y_preprocessed = self._apply_preprocessing(x_grad, y=_y, fit=False, no_grad=False)
            logits = self._model(inputs_t)[-1]  # -1 refers to the last layer output, i.e., logits
            if _y is None:
                return logits

            return logits, self.reduce_labels(y_preprocessed)

        # Compute averaged logits
        averaged_logits, labels_t = get_logits(_y=y_grad)
        for _ in range(nb_samples - 1):
            averaged_logits += get_logits()
        averaged_logits /= nb_samples

        # Compute the gradient and return
        loss = self._loss(averaged_logits, labels_t)
        self._model.zero_grad()
        loss.backward()
        grads = x_grad.grad

        assert grads.shape == x.shape

        return grads

    return types.MethodType(loss_gradient, instance)
