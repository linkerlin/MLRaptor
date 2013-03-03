"""
The :mod:`mix` module gathers point-estimate and variational approximations
 to standard parametric mixture modeling
"""

from .MixModel import MixModel
from .QMixModel import QMixModel
from .QDPMixModel import QDPMixModel
from ..util.MLUtil import logsumexp

__all__ = ['logsumexp', 'MixModel', 'QMixModel', 'QDPMixModel']
