import multiprocessing as mp
import ctypes
import numpy as np
import scipy.linalg
import time

info = mp.get_logger().info

#================================================================== Global Data
def Estep_parallel( Xin, w, MuList, SigmaList, nProc=2, chunksize=1, doVerbose=False ):
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
  if doVerbose:
    logger = mp.log_to_stderr()
    logger.setLevel( logging.INFO )

  global X
  X = Xin

  GMMCompIterator = [ (k,MuList[k],SigmaList[k]) for k in xrange( len(MuList) )]
  mypool = mp.Pool( processes=nProc )
  myParOp = mypool.map_async( loggausspdf_globaldata, GMMCompIterator, chunksize=chunksize )
  resultList = myParOp.get()
  #st = time.time()
  logResp = np.vstack(resultList)
  #print '  Reduction: %.2f sec' % (time.time()-st)
  # Time to agg results into single matrix: 0.07 sec for N=250000,K=25
  # Conclusion: agg results takes almost no time relative to each individual job
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

  D = Sigma.shape[1]
  distPerRow, cholSigma = distMahal_globaldata( mu, Sigma )
  logdetSigma = 2*np.sum( np.log( np.diag(cholSigma) ) )
  logNormConst = -0.5*D*np.log(2*np.pi) - 0.5*logdetSigma
  logpdfPerRow = logNormConst - 0.5*distPerRow
  return logpdfPerRow
