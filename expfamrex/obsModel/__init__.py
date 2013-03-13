"""
The :mod:`obsModel` module provides standard parametric distributions
  in the exponential family.
"""

from .GaussianDistr import GaussianDistr
from .WishartDistr import WishartDistr
from .GaussWishDistrIndep import GaussWishDistrIndep

from .BernoulliDistr import BernoulliDistr
from .BetaDistr import BetaDistr
from .BernObsCompSet import BernObsCompSet

from .GaussObsCompSet import GaussObsCompSet

__all__ = ['GaussianDistr', 'GaussWishDistrIndep', 'WishartDistr', \
            'BernoulliDistr', 'BetaDistr', 'BernObsCompSet', \
           'GaussObsCompSet']
