#! /usr/bin/python -W ignore::RuntimeWarning
"""
Gaussian Mixture Models.
 with online learning via EM (Max Likelihood approach... no priors)
"""
import numpy as np
import multiprocessing as mp
import ctypes
import time
import argparse
import sklearn.cluster
import sklearn.mixture
from sklearn.utils.extmath import logsumexp, pinvh
import glob
import logging
import random

np.set_printoptions( precision=2, suppress=False )

T_START = time.time()
info = mp.get_logger().info

EPS = np.finfo(float).eps
MIN_VAR = 0.01

D = 10  #Temporary!
K = 3

delay = 1
kappa = 0.5

shLock = mp.Lock()
shIterCount = mp.Array( ctypes.c_double, 1, lock=shLock )
shWeights = None
shMeans = None
shCovars  = None 


def sh2np( shmemArr, targetshape=None ):
  A = np.ctypeslib.as_array( shmemArr.get_obj() )
  if targetshape is not None:
    return A.reshape( targetshape )
  else:
    return A
    
def log_multivariate_normal_density_diag(X, Mu, Sigma):
    """Compute Gaussian log-density at X for a diagonal model
       Parameters:
         X : array_like, N x D
         Mu: array_like, K x D
         Sigma: array_like, K x D 
       Returns:  
         lpr : NxK matrix
                        where lpr[n,k] = Pr( X[n,:] | mu[k,:], sigma[k,:] )
    """
    N, D = X.shape
    lpr = np.dot( X**2, (1.0/Sigma).T ) # x^T invS x term
    lpr -= 2*np.dot( X, (Mu/Sigma).T )  # x^T invS mu term
    lpr += np.sum( Mu**2/Sigma, axis=1) # mu^T invS mu term
    lpr += D * np.log(2*np.pi)  # norm constants
    lpr += np.sum(np.log(Sigma), axis=1)
    return -0.5*lpr    

def fitChunkFromFile( fname ):
    fitChunk( np.loadtxt(fname)  )
  
def fitChunk( Xchunk ):
    N,D = Xchunk.shape
    
    shLock.acquire()
    w = sh2np( shWeights ).copy()
    m = sh2np( shMeans  , (K,D) ).copy()
    c = sh2np( shCovars , (K,D) ).copy()
    shLock.release()
     
    # E step
    #  resp : N x K matrix
    #         resp[n,k] = Pr( z[n]=k | X[n], Mu[k], Sigma[k] )
    #         properly normalized, so sum( resp, axis=1) = 1.0
    lpr = np.log(w) \
              + log_multivariate_normal_density_diag(Xchunk, m, c)
    lprNORMCONST = logsumexp(lpr, axis=1)
    resp = np.exp(lpr - lprNORMCONST[:, np.newaxis])
      
    # M step    
    Nresp = resp.sum(axis=0)
    wChunk = Nresp / ( Nresp.sum() + EPS )
    
    wavg_X = np.dot(resp.T, Xchunk)
    mChunk = wavg_X / (Nresp[:,np.newaxis] )

    wavg_X2 = np.dot(resp.T, Xchunk**2)
    wavg_M2 = m**2 * Nresp[:,np.newaxis] 
    wavg_XM = wavg_X * m
    
    cChunk = wavg_X2 -2*wavg_XM + wavg_M2
    cChunk /= Nresp[:,np.newaxis]
    #avg_X2 = np.dot(resp.T, Xchunk * Xchunk) * (N*wChunk[:,np.newaxis] )
    #avg_means2 = m ** 2
    #avg_X_means = m * weighted_X_sum * (N*wChunk[:,np.newaxis] )
    #cChunk = avg_X2 - 2 * avg_X_means + avg_means2 + MIN_VAR
    
    # Synchronize global 
    shLock.acquire()
    tstart = time.time()- T_START
    ww = sh2np( shWeights )
    
    #info("   used to compute local updates %.3f %.3f" % ( w[0], w[1] ) )
    #info("now using possibly fresher value %.3f %.3f" % ( ww[0], ww[1] ) )

    mm = sh2np( shMeans  , (K,D) )
    cc = sh2np( shCovars , (K,D) )
    t   = sh2np( shIterCount, (1,1) )
    t += 1
    
    rho = (t + delay)**(-kappa)
    ww[:] = (1-rho)*ww + rho*wChunk
    mm[:,:] = (1-rho)*mm + rho*mChunk
    cc[:,:] = (1-rho)*cc + rho*cChunk
    
    tstop = time.time() - T_START
    
    #info("                                 %.3f | %.4f-%.4f sec" % ( rho, tstart, tstop ) )
    shLock.release()
    
      
   
def initShared( Xinit ):  
    D = Xinit.shape[1] 
    #info("Creating shared arrays | K=%d, D=%d" % (K,D) )

    t = sh2np( shIterCount )
    t = 1 # Already processed first chunk
    
    w = sh2np( shWeights )
    w[:] = 1./K
    
    m = sh2np( shMeans, (K,D) )
    m[:,:] = sklearn.cluster.KMeans( K ).fit(Xinit).cluster_centers_
    
    c = sh2np( shCovars, (K,D) )
    tied_cv = np.cov(Xinit.T) + MIN_VAR * np.eye(D)
    c[:,:] = np.tile(np.diag(tied_cv), (K, 1)) 
    
def par_fit( K, ChunkGenerator, nProc=1, timeout=120, doVerbose=False): 
    logger = mp.log_to_stderr()
    if doVerbose:
      logger.setLevel(logging.INFO)
       
    item1 = ChunkGenerator.next()
    if type( item1 ) == str:
      Xinit = np.loadtxt( item1 ) 
      funcChunk = fitChunkFromFile
    else:
      funcChunk = fitChunk
      Xinit = item1
    D = Xinit.shape[1]
    initShared( Xinit )
    
    # Run in parallel!        
    workerpool = mp.Pool( processes=nProc )
    R = workerpool.map_async( funcChunk, ChunkGenerator )

    R.get( timeout=timeout )  
    
    workerpool.close() # just in case...
    
    # Build mixture model with resulting parameters!
    mygmm = sklearn.mixture.GMM( n_components=K )

    mygmm.weights_ = sh2np( shWeights )
    mygmm.covars_ = sh2np( shCovars, (K,D) )
    mygmm.means_ = sh2np( shMeans, (K,D) )
    return mygmm

def FileChunkGenerator( nRep=1, datapathpattern='/Users/mhughes/data/K3D10/*.dat' ):
    fList = glob.glob( datapathpattern )
    for rep in xrange( nRep ):
        random.shuffle(fList)
        for fname in fList:
            yield fname

def DataChunkGenerator( datapathpattern='/Users/mhughes/data/K3D10/*.dat' ):
    fList = glob.glob( datapathpattern )
    for rep in xrange( nRep ):
        random.shuffle(fList)
        for fname in fList:
            yield np.loadtxt(fname)
      
def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument( 'datapath', type=str )
    Parser.add_argument( '-R', '--nRep', type=int, default=1 )
    Parser.add_argument( '-K', '--nComp', type=int, default=3 )
    Parser.add_argument( '--kappa', type=float, default=0.5 )
    Parser.add_argument( '--delay', type=float, default=1 )
    Parser.add_argument( '--nProc', type=int, default=1 )
    Parser.add_argument( '-v', '--doVerbose', action='store_true', default=False )
    Parser.add_argument( '--doPrint', action='store_true', default=True )
    args = Parser.parse_args()
    kappa = args.kappa
    delay = args.delay
    K = args.nComp
    
    # Reallocate shared memory
    DGtemp = FileChunkGenerator(args.nRep,  args.datapath)
    Xexample =  np.loadtxt( DGtemp.next() )
    D = Xexample.shape[1]
    
    global shWeights, shMeans, shCovars
    shWeights = mp.Array( ctypes.c_double, K, lock=shLock )
    shMeans = mp.Array( ctypes.c_double, K*D, lock=shLock )
    shCovars  = mp.Array( ctypes.c_double, K*D, lock=shLock )  
    
    tstart = time.time()
    DG = FileChunkGenerator(args.nRep,  args.datapath)
    mygmm = par_fit( K, DG, nProc=args.nProc, doVerbose=args.doVerbose )
    tstop = time.time()
    
    print 'Processed %d pages of data after %.1f sec.' % (sh2np(shIterCount,1), tstop-tstart)
    
    if args.doPrint:
      print 'w:', mygmm.weights_
      print 'MU' 
      for k in range(K):
        print ' '.join([ '% 5.1f'%(x) for x in mygmm.means_[k,:]] )
      print 'Sigma'
      for k in range(K):
        print np.diag( mygmm.covars_[k,:] )
    
    return mygmm
    
if __name__ == '__main__':
    main()
