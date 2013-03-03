"""
The :mod:`obsModel` module provides standard parametric distributions
  in the exponential family.
"""

from .GaussDistr import GaussDistr
from .GaussWishDistr import GaussWishDistr
from .BernoulliDistr import BernoulliDistr

from .GaussianObsCompSet import GaussianObsCompSet

__all__ = ['GaussDistr', 'GaussWishDistr', 'BernoulliDistr', \
           'GaussianObsCompSet']