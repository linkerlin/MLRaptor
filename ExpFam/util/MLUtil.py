import numpy as np

def flatstr2np( xvecstr ):
  return np.asarray( [float(x) for x in xvecstr.split()] )

def np2flatstr( X, fmt="% .6f" ):
  return ' '.join( [fmt%(x) for x in X.flatten() ] )  

def logsoftev2softev( logSoftEv, axis=1):
  lognormC = np.max( logSoftEv, axis)
  if axis==0:
    logSoftEv = logSoftEv - lognormC[np.newaxis,:]
  elif axis==1:
    logSoftEv = logSoftEv - lognormC[:,np.newaxis]
  SoftEv = np.exp( logSoftEv )
  return SoftEv, lognormC

def logsumexp( logA, axis=None):
  logA = np.asarray( logA )

  #if axis is None and logA.ndim==1:
  #  logA = logA.reshape( (logA.size,1) )

  # Fix logA's dynamic range, so that
  #  largest entry is 0, all others smaller than 0
  # i.e.   [-5 -100 -1] -->  [-4 -99  0]
  #        [ 6  10  1 ] -->  [-4   0 -9]
  logAmax = logA.max( axis=axis )

  if axis is None:
    logA = logA - logAmax
  elif axis==1:
    logA = logA - logAmax[:,np.newaxis]
  elif axis==0:
    logA = logA - logAmax[np.newaxis,:]

  assert np.allclose( logA.max(), 0.0 )
  
  logA = np.log( np.exp( logA ).sum( axis=axis )  )
  return logA + logAmax
