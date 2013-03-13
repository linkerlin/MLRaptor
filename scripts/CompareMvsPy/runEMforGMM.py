'''
Run EM algorithm for fitting Gaussian mixture model with K components

Author: Mike Hughes (mike@michaelchughes.com)

Usage
--------
Run as a script at the command line
>> python runEMforGMM.py datafilename K Niter savefilename seed

Arguments
--------
   datafilename  :  string path to .MAT file containing
                      observed data matrix X (each row is iid observation)

   K             :  integer # of components in mix model
   Niter         :  max # of iterations to run
   savefilename  :  string path to .MAT file to save final mixture model
                       with fields .w, .mu, and .Sigma
   seed          :  integer seed for random number generation (used for init only)
'''

import argparse
import time
import random
import numpy as np
import scipy.linalg
import scipy.io

CONVERGE_THR = 1e-6
MIN_COVAR    = 1e-8
EPS          = 1e-15

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument( 'datafilename' )
  parser.add_argument( 'K', type=int )
  parser.add_argument( 'Niter', type=int )
  parser.add_argument( 'savefilename', type=str)
  parser.add_argument( 'seed', type=int )
  parser.add_argument( '--MIN_COVAR', type=float, default=MIN_COVAR )
  return parser.parse_args()

def main( datafilename, K, Niter, savefilename, seed ):
  X = scipy.io.loadmat( datafilename )['X']
  loglik = -np.inf*np.ones( Niter )

  Resp = init_responsibilities( X, K, seed)
  tstart = time.time()
  for t in xrange( Niter ):
    model = Mstep( X, Resp )
    Resp, loglik[t] = Estep( X, model )
    
    print '%5d/%d after %.0f sec | %.8e' % (t+1, Niter, time.time()-tstart, loglik[t])    
    deltaLogLik = loglik[t] - loglik[t-1]
    if deltaLogLik < CONVERGE_THR:
      break
    if deltaLogLik < 0:
      print 'WARNING: loglik decreased!'
  scipy.io.savemat( savefilename, model, oned_as='row' ) # oned_as kwarg avoid stupid warning
  return model, loglik

def init_responsibilities( X, K, seed):
  N,D = X.shape
  random.seed( seed )
  rowIDs = random.sample( xrange(N), K ) #without replacement
  mu = X[rowIDs, : ]

  logResp = np.zeros( (N, K) )
  for k in xrange( K ):
    logResp[:,k] = loggausspdf( X, mu[k,:], np.eye(D) )
  logPrPerRow = logsumexp( logResp, axis=1 )
  Resp = np.exp( logResp - logPrPerRow[:,np.newaxis] )
  return Resp

###################################################
def Estep(X, model):
  w = model['w']
  mu = model['mu']
  Sigma = model['Sigma']

  N = X.shape[0]
  K = mu.shape[0]
  logResp = np.zeros( (N, K) )
  for k in xrange( K ):
    logResp[:,k] = loggausspdf( X, mu[k,:], Sigma[:,:,k] )
  logResp += np.log( w )

  logPrPerRow = logsumexp( logResp, axis=1 )
  Resp = np.exp( logResp - logPrPerRow[:,np.newaxis] )
  return Resp, np.sum(logPrPerRow)
  
def Mstep(X, Resp):
  N,D = X.shape
  K = Resp.shape[1]

  Nk = np.sum( Resp, axis=0) + EPS
  w  = Nk/N
  mu = np.dot( Resp.T, X ) / Nk[:,np.newaxis]
  Sigma = np.zeros( (D,D,K) )
  for k in xrange( K ):
    Xdiff = X - mu[k,:]
    Xdiff = Xdiff * np.sqrt( Resp[:,k] )[:,np.newaxis]
    Sigma[:,:,k] = np.dot( Xdiff.T, Xdiff) / Nk[k] + MIN_COVAR*np.eye(D)
  return dict( w=w, mu=mu, Sigma=Sigma )

###################################################
def loggausspdf( X, mu, Sigma):
  ''' Calc log p( x | mu, Sigma) for each row of matrix X
  '''
  N,D = X.shape
  dist, cholSigma = distMahal( X, mu, Sigma )
  logdetSigma = 2*np.sum( np.log( np.diag(cholSigma) ) )
  logNormConst = -0.5*D*np.log(2*np.pi) - 0.5*logdetSigma
  logpdfPerRow = logNormConst - 0.5*dist
  return logpdfPerRow
  
def distMahal( X, mu, Sigma ):
  ''' Calc mahalanobis distance: (x-mu)^T Sigma^{-1} (x-mu)
       for each row of matrix X
  '''
  N,D = X.shape
  Xdiff = X - mu
  cholSigma = scipy.linalg.cholesky( Sigma, lower=True)
  Q = scipy.linalg.solve_triangular( cholSigma, Xdiff.T, lower=True )
  distPerRow = np.sum( Q**2, axis=0 )
  return distPerRow, cholSigma


def logsumexp( logA, axis=None):
  ''' Compute log( sum(exp(logA))) in numerically stable way
  '''
  logA = np.asarray( logA )
  logAmax = logA.max( axis=axis )

  if axis is None:
    logA = logA - logAmax
  elif axis==1:
    logA = logA - logAmax[:,np.newaxis]
  elif axis==0:
    logA = logA - logAmax[np.newaxis,:]
  assert np.allclose( logA.max(), 0.0 )
  logA = np.log( np.sum( np.exp( logA ), axis=axis )  )
  return logA + logAmax


if __name__ == '__main__':
  args = parse_args()
  MIN_COVAR = args.MIN_COVAR
  main(  args.datafilename, args.K, args.Niter, args.savefilename, args.seed )
