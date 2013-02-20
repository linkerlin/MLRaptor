"""
The :mod:`learn` module provides standard learning algorithms
  such as EM and Variational Bayesian
"""

from .LearnAlg import LearnAlg
from .EMLearnAlg import EMLearnAlg
from .VBLearnAlg import VBLearnAlg

__all__ = ['LearnAlg', 'EMLearnAlg', 'VBLearnAlg']
