"""
The :mod:`mix` module gathers point-estimate and variational approximations
   for Bayesian mixture modeling, both parametric and nonparametric (via Dir. Process)
"""

from .MixModel import MixModel
from .DPMixModel import DPMixModel
from ..util.MLUtil import logsumexp

__all__ = ['logsumexp', 'MixModel', 'DPMixModel']
