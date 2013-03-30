import multiprocessing as mp
import ctypes
import argparse
import numpy as np
import scipy.linalg
import time
import logging

info = mp.get_logger().info

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
  distPerRow, cholSigma = distMahal( X, mu, Sigma )
  logdetSigma = 2*np.sum( np.log( np.diag(cholSigma) ) )
  logNormConst = -0.5*D*np.log(2*np.pi) - 0.5*logdetSigma
  logpdfPerRow = logNormConst - 0.5*distPerRow
  return logpdfPerRow

#================================================================== Global Data
def Estep_parallel( X, w, MuList, SigmaList, nProc=2, chunksize=1 ):
  ''' Returns
      -------
        logResp : N x K vector of log posterior probabilities 

                  logResp[n,k] : n-th data point's posterior under k-th comp
  '''
  '''def obs_comp_generator( MuList, SigmaList):
    for k in xrange( len(MuList) ):
      yield k,MuList[k], SigmaList[k]
  GMMCompIterator = obs_comp_generator( MuList, SigmaList )
  '''
  GMMCompIterator = [ (k,MuList[k],SigmaList[k]) for k in xrange( len(MuList) )]
  mypool = mp.Pool( processes=nProc )
  myParOp = mypool.map_async( loggausspdf_globaldata, GMMCompIterator, chunksize=chunksize )
  resultList = myParOp.get()
  #st = time.time()
  logResp = np.vstack(resultList)
  #print '  Reduction: %.2f sec' % (time.time()-st)  # Time to agg results into single matrix: 0.07 sec for N=250000,K=25
  return logResp.T + np.log(w)

def distMahal_globaldata( mu, Sigma):
  Xdiff = X - mu
  cholSigma = scipy.linalg.cholesky( Sigma, lower=True)
  Q = np.linalg.solve( cholSigma, Xdiff.T )
  distPerRow = np.sum( Q**2, axis=0 )
  return distPerRow, cholSigma

def loggausspdf_globaldata( msTuple ):
  ''' Calc log p( x | mu, Sigma) for each row of global matrix X
  '''
  k,mu,Sigma = msTuple
  msg = "Computing logPDF for component %d" % (k)
  info( msg )

  distPerRow, cholSigma = distMahal_globaldata( mu, Sigma )
  logdetSigma = 2*np.sum( np.log( np.diag(cholSigma) ) )
  logNormConst = -0.5*D*np.log(2*np.pi) - 0.5*logdetSigma
  logpdfPerRow = logNormConst - 0.5*distPerRow
  return logpdfPerRow


#================================================================== Main Function
if __name__ == '__main__':
  np.random.seed( 8675309)
  Parser = argparse.ArgumentParser()

  Parser.add_argument( '--nTrial', type=int, default=1)
  Parser.add_argument( '--nProc', type=int, default=1)
  Parser.add_argument( '--chunksize', type=int, default=1)
  Parser.add_argument( '-v', '--doVerbose', action='store_true', default=False)
  Parser.add_argument( '--K', type=int, default=25)
  Parser.add_argument( '--D', type=int, default=64)
  Parser.add_argument( '--N', type=int, default=50000)
  args=Parser.parse_args()
  N=args.N;  K=args.K;  D=args.D

  X = 10*np.random.randn( N, D )

  w = np.ones( K )/float(K)  
  MuList = 5*np.random.randn( K, D)
  SigmaList = .25*np.random.rand( K, D, D)
  for k in xrange(K):
    SigmaList[k] += np.eye(D)
    SigmaList[k] = np.dot( SigmaList[k].T, SigmaList[k] )
  
  if args.doVerbose:
    logger = mp.log_to_stderr()
    logger.setLevel( logging.INFO )

  # ---------------------------- Execute algorithm
  print "Running GMM Estep with %d processes" % (args.nProc)
  stime = time.time()
  for trial in xrange( args.nTrial ):
    if args.nProc == 1:
      logResp = Estep_serial( X, w, MuList, SigmaList )
    else:
      logResp = Estep_parallel( X, w, MuList, SigmaList, nProc=args.nProc, chunksize=args.chunksize )
  etime = time.time() - stime
  print "%.3f sec after %d trials" % (etime, args.nTrial)
  print logResp
