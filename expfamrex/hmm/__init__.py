"""
The :mod:`admix` module provides variational approximations to 
  admixture models (in the style of Latent Dirichlet Allocation).
"""

from .HMM import HMM
import HMMUtil

__all__ = ['HMM', 'HMMUtil']