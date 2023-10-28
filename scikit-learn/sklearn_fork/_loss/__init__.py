"""
The :mod:`sklearn_fork._loss` module includes loss function classes suitable for
fitting classification and regression tasks.
"""

from .loss import (
    HalfSquaredError,
    AbsoluteError,
    PinballLoss,
    HuberLoss,
    HalfPoissonLoss,
    HalfGammaLoss,
    HalfTweedieLoss,
    HalfTweedieLossIdentity,
    HalfBinomialLoss,
    HalfMultinomialLoss,
)


__all__ = [
    "HalfSquaredError",
    "AbsoluteError",
    "PinballLoss",
    "HuberLoss",
    "HalfPoissonLoss",
    "HalfGammaLoss",
    "HalfTweedieLoss",
    "HalfTweedieLossIdentity",
    "HalfBinomialLoss",
    "HalfMultinomialLoss",
]
