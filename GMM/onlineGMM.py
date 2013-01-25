#! /usr/bin/python -W ignore::DeprecationWarning
"""
Gaussian Mixture Models.
 with online learning via EM (Max Likelihood approach... no priors)
"""
import numpy as np
import time
import argparse
import sklearn.cluster
import sklearn.mixture
from sklearn.utils.extmath import logsumexp, pinvh
import os.path
import sys

EPS = np.finfo(float).eps

def np2flatstr( X ):
  return ', '.join( [str(x) for x in X.flatten() ] )  

class onlineGMM(object):
  '''
  '''
  
  def __init__( self, n_components=1, kappa=0.5, delay=1.0, covariance_type='diag', min_covar=0.01):
    self.n_components = n_components
    self.K = n_components
    self.covariance_type = covariance_type
    self.min_covar = min_covar
    self.kappa = kappa
    self.delay = delay
    
    self.needInitFlag = True

  def fit( self, DataGenerator, random_state, initname='kmeans', savefilename='GMMtrace.dat', saveEvery=5, printEvery=1 ):
    self.initname = 'kmeans'
    self.savefilename = savefilename
    self.iterCount = 0
    self.start_time = time.time()
    self.saveEvery=saveEvery
    self.printEvery=printEvery
    for iterCount, Xchunk in enumerate(DataGenerator):
      if iterCount==0 and self.needInitFlag:
        self.init_params( Xchunk, random_state )
        evBound = self.fit_chunk( Xchunk, iterCount )
      else:
        evBound = self.fit_chunk( Xchunk, iterCount )
      self.save_state()
      self.print_state(evBound)
      self.iterCount += 1
    self.print_state(evBound, doFinal=True)
    
  def fit_chunk( self, Xchunk, iterCount, burnTHR=0):
    if iterCount < burnTHR:
      rho = 0.5*( .99**( iterCount ) )
    else:
      rho = float(iterCount-burnTHR + self.delay)**(-self.kappa)
    resp, evBound = self.E_step( Xchunk )
    self.M_step( Xchunk, resp, rho )
    return evBound
    
  def E_step( self, Xchunk ):
    '''
      Returns resp : N x K matrix
             resp[n,k] = Pr( z[n]=k | X[n], Mu[k], Sigma[k] )
             properly normalized, so sum( resp, axis=1) = 1.0
    '''
    lpr = np.log( self.w ) \
              + log_multivariate_normal_density_diag(Xchunk,self.Mu,self.Sigma)
    lprSUM = logsumexp(lpr, axis=1)
    evidenceBound = lprSUM.sum()
    resp = np.exp(lpr - lprSUM[:, np.newaxis])
    assert np.allclose( resp.sum(axis=1), 1.0 )
    return resp, evidenceBound
  
  def M_step( self, Xchunk, resp, rho=1.0 ):
    '''
       Updates internal mixture model parameters
       Returns: nothing
    '''
    Nresp = resp.sum(axis=0)
    wChunk = Nresp / ( Nresp.sum() + EPS )
    assert np.allclose(1.0, wChunk.sum())
      
    wavg_X = np.dot(resp.T, Xchunk)
    muChunk = wavg_X / (Nresp[:,np.newaxis] + EPS)

    wavg_X2 = np.dot(resp.T, Xchunk**2)
    wavg_M2 = muChunk**2 * Nresp[:,np.newaxis] 
    wavg_XM = wavg_X * muChunk
    sigChunk = wavg_X2 -2*wavg_XM + wavg_M2
    sigChunk /= (Nresp[:,np.newaxis] + EPS)
    sigChunk += self.min_covar
    
    '''
    sigChunk = np.zeros( muChunk.shape )
    for k in range( muChunk.shape[0] ):
      S = np.zeros( Xchunk.shape[1] )
      for n in range( Xchunk.shape[0] ):
        S += resp[n,k]*( Xchunk[n,:] - muChunk[k,:] )**2
      sigChunk[k] = S/Nresp[k]
    '''
    #k = np.argmin( self.Mu[:,0].flatten() )
    #print rho, self.Sigma[k,0],  sigChunk[k,0]
        
    self.w = (1-rho)*self.w + rho*wChunk
    self.Mu = (1-rho)*self.Mu + rho*muChunk
    self.Sigma = (1-rho)*self.Sigma + rho*sigChunk
    
  def init_params( self, Xchunk, seed=101):
    print Xchunk.mean()
    D = Xchunk.shape[1] 
    K = self.K    
    self.w = 1./K * np.ones( K )
    self.Mu = sklearn.cluster.KMeans( K, random_state=seed ).fit(Xchunk).cluster_centers_
    #tied_cv = np.cov(Xchunk.T) + self.min_covar*np.eye(D)
    #self.Sigma = np.tile(np.diag(tied_cv), (K, 1)) 
    self.Sigma = np.ones( (K,D) )
    
  def print_state( self, evBound, doFinal=False ):
    if self.iterCount==0:
      print 'Initialized via %s.' % (self.initname)
    elif self.iterCount % self.printEvery==0:
      print '  %5d batches after %6.0f sec. | evidence % .3e' % (self.iterCount, time.time()-self.start_time, evBound)
    elif doFinal:
      print '  %5d batches after %6.0f sec. Done.' % (self.iterCount, time.time()-self.start_time)
    
  def save_state( self ):
    if self.iterCount==0: 
      mode = 'w'
    else:
      mode = 'a'
    if self.iterCount % (self.saveEvery)==0:
      filename, ext = os.path.splitext( self.savefilename )
      fid = open( filename+'.w', mode)
      fid.write( np2flatstr(self.w) + '\n' )
      fid.close()
      fid = open( filename+'.mu', mode)
      fid.write( np2flatstr( self.Mu ) + '\n' )
      fid.close()
      fid = open( filename+'.sigma', mode)
      fid.write( np2flatstr( self.Sigma) + '\n' )
      fid.close()
      fid = open( filename+'.batchID', mode)
      fid.write( str(self.iterCount) + '\n' )
      fid.close()
    
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

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument( 'datagenModule', type=str )
    Parser.add_argument( '-K', '--nComp', type=int, default=3 )
    Parser.add_argument( '--batch_size', type=int, default=500 )
    Parser.add_argument( '--nBatch', type=int, default=50 )
    Parser.add_argument( '--nRep', type=int, default=1 )
    
    Parser.add_argument( '--kappa', type=float, default=0.5 )
    Parser.add_argument( '--delay', type=float, default=1 )
    Parser.add_argument( '--seed', type=int, default=8675309 )
    
    Parser.add_argument( '-v', '--doVerbose', action='store_true', default=False )
    Parser.add_argument( '--doPrint', action='store_true', default=False )
    args = Parser.parse_args()
        
    # Dynamically load module provided by user as data-generator
    #   this must implement a generator function called "minibatch_generator"
    datagenmod = __import__( args.datagenModule, fromlist=[])
    DataGen = datagenmod.minibatch_generator( args.batch_size, args.nBatch, args.nRep, args.seed )    
        
    if 'print_data_info' in dir( datagenmod ):
      datagenmod.print_data_info()
      
    mygmm = onlineGMM( n_components=args.nComp, kappa=args.kappa, delay=args.delay, min_covar=0.1 )
    mygmm.fit( DataGen, args.seed )
    
    if args.doPrint:
    
      np.set_printoptions( precision=2, suppress=False )

      sortIDs = np.argsort( mygmm.Mu[:,0].flatten() )
      print 'w:', mygmm.w[ sortIDs ]
      print 'Mu' 
      for k in sortIDs:
        print ' '.join([ '% 5.1f'%(x) for x in mygmm.Mu[k,:]] )
      print 'Sigma'
      for k in sortIDs:
        print np.diag( mygmm.Sigma[k,:] )
    
    return mygmm
    
if __name__ == '__main__':
    main()
