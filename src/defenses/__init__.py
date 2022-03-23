from src.defenses.base import InstanceFunction, InstancePreprocessorPyTorch, bpda_identity
from src.defenses.compression import Quantization, DCT, JPEG
from src.defenses.ensemble import Ensemble
from src.defenses.perturbation import (
    FFTPerturbation,
    NoiseInjection,
    NoiseInjectionPyTorch,
    GaussianNoisePyTorch,
    GaussianNoisePyTorchNoClip,
)
from src.defenses.transformation import Gaussian, Median, Swirl, ResizePad, Crop
