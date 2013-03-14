"""
The :mod:`admix` module provides variational approximations to 
  admixture models (in the style of Latent Dirichlet Allocation).
"""

from .HMM import HMM
from .HMMUtil import *

__all__ = ['HMM', 'HMMUtil']
