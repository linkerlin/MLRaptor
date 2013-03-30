import numpy as np
import scipy.linalg

#================================================================== Standard data as arg
def Estep_serial( X, w, MuList, SigmaList ):
  N = X.shape[0]
  K = w.shape[0]
  logResp = np.zeros( (N, K) )
  for k in xrange( K ):
    logResp[:,k] = loggausspdf( X, MuList[k,:], SigmaList[k,:,:] )
  logResp += np.log( w )
  return logResp

def distMahal( X, mu, Sigma ):
  ''' Calc mahalanobis distance: (x-mu)^T Sigma^{-1} (x-mu)
       for each row of matrix X
  '''
  Xdiff = X - mu
  cholSigma = scipy.linalg.cholesky( Sigma, lower=True)
  Q = np.linalg.solve( cholSigma, Xdiff.T )
  distPerRow = np.sum( Q**2, axis=0 )
  return distPerRow, cholSigma

def loggausspdf( X, mu, Sigma):
  ''' Calc log p( x | mu, Sigma) for each row of matrix X
  '''
  D = X.shape[1]
  distPerRow, cholSigma = distMahal( X, mu, Sigma )
  logdetSigma = 2*np.sum( np.log( np.diag(cholSigma) ) )
  logNormConst = -0.5*D*np.log(2*np.pi) - 0.5*logdetSigma
  logpdfPerRow = logNormConst - 0.5*distPerRow
  return logpdfPerRow

