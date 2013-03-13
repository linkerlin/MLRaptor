"""
The :mod:`util` module gathers utility calculations like "logsumexp"
"""

from .MLUtil import logsumexp, np2flatstr, flatstr2np

__all__ = ['logsumexp', 'np2flatstr', 'flatstr2np']
