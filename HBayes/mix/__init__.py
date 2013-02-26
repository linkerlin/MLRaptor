"""
The :mod:`mix` module gathers point-estimate and variational approximations
 to standard parametric mixture modeling
"""

from .MixModel import MixModel
from .QMixModel import QMixModel
from .QDPMixModel import QDPMixModel
from .GMM import GMM
from .QGMM import QGMM
from .QDPGMM import QDPGMM
from ..util.MLUtil import logsumexp

__all__ = ['logsumexp', 'MixModel', 'QMixModel', 'GMM', 'QGMM', 'QDPGMM']
