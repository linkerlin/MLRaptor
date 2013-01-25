import numpy as np
try:
  from scipy.misc import logsumexp as _logsumexp
except Exception:
  from scipy.maxentropy import logsumexp as _logsumexp



def logsumexp(a, axis=None):
  try:
    _logsumexp( a, axis)
  except Exception:
    if axis is None:
      # Use the scipy.maxentropy version.
      return _logsumexp(a)
    a = np.asarray(a)
    shp = list(a.shape)
    shp[axis] = 1
    a_max = a.max(axis=axis)
    s = np.log( np.exp(a - a_max.reshape(shp)).sum(axis=axis))
    lse  = a_max + s
    return lse
