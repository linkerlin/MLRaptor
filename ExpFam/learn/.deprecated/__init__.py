"""
The:mod:`learn'  modulerovides standard learning algorithms
  such as EM and Variational Bayesian
"""
from .LearnAlg import LearnAlg
from .EMLearnAlg import EMLearnAlg
from .VBLearnAlg import VBLearnAlg
from .OnlineEMLearnAlg import OnlineEMLearnAlg
from .OnlineVBLearnAlg import OnlineVBLearnAlg

__all__ = ['LearnAlg', 'EMLearnAlg', 'VBLearnAlg','OnlineEMLearnAlg',\
            'OnlineVBLearnAlg']
