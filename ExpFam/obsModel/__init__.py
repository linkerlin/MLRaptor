"""
The :mod:`obsModel` module provides standard parametric distributions
  in the exponential family.
"""

from .GaussianDistr2 import GaussianDistr2
from .WishartDistr import WishartDistr
from .GaussWishDistrIndep import GaussWishDistrIndep

from .GaussDistr import GaussDistr
from .GaussWishDistr import GaussWishDistr
from .BernoulliDistr import BernoulliDistr

from .GaussianObsCompSet import GaussianObsCompSet
from .GaussObsCompSet2 import GaussObsCompSet2

__all__ = ['GaussDistr', 'GaussWishDistr', 'BernoulliDistr', \
           'WishartDistr', 'GaussianDistr2', 'GaussWishDistrIndep',
           'GaussianObsCompSet', 'GaussObsCompSet2']
