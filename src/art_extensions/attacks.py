import torch
from art.attacks.evasion import ProjectedGradientDescentPyTorch as PGD


class AggMoPGD(PGD):
    """
    Experimental implementation of AggMo PGD.

    Extra Args:
       b: number of moments in AggMo.

    TODO: It is unclear how the step size is defined per PGD iteration.

    | Paper link: https://openreview.net/pdf?id=p4SrFydwO5
    """

    def __init__(self, *args, **kwargs):
        self.b = kwargs.pop('b')
        super().__init__(*args, **kwargs)

    def _generate_batch(self, x: torch.Tensor, *args, **kwargs):
        """Initialize AggMo variables when starting handling a batch, and let super class generate the perturbation.
        """
        batch_size = x.shape[0]
        self.v = torch.zeros_like(x).repeat(self.b, 1, 1, 1, 1).to(self.estimator.device)
        self.mu = 1 - 0.1 ** torch.arange(self.b).to(self.estimator.device)
        self.mu = self.mu.repeat(batch_size, 1).T
        return super()._generate_batch(x, *args, **kwargs)

    def _apply_perturbation(self, x: torch.Tensor, perturbation: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Generate new perturbation by AggMo, and let super class apply it.
        """
        with torch.no_grad():
            self.v = self.mu[..., None, None, None] * self.v + perturbation
            new_perturbation = self.v.mean(dim=0)

        return super()._apply_perturbation(x, new_perturbation, *args, **kwargs)
