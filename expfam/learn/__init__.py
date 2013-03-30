"""
The:mod:`learn'  modulerovides standard learning algorithms
  such as EM and Variational Bayesian
"""
from .LearnAlg import LearnAlg
from .VBLearnAlg import VBLearnAlg
from .OnlineVBLearnAlg import OnlineVBLearnAlg

from .VBInferHeldout import VBInferHeldout

__all__ = ['LearnAlg', 'VBLearnAlg','OnlineVBLearnAlg', 'VBInferHeldout']
