"""
Expectation Maximization learning algorithm
  for a Gaussian Mixture Model (GMM)

Author: Mike Hughes (mike@michaelchughes.com)

Allows:
  * a variety of initializations (default=kmeans)
  * easy logging of parameter values/likelihood traces across iterations
     (stored to file in results/ directory)  

Usage
-------
 To fit a 5-component on data matrix X (each row is an obs. vector)

  mygmm = GMM( K=5, covar_type='diag')
  em    = EMLearnerGMM( mygmm, Niter=100 )
  em.fit( X )

Related
-------
  GMM.py

References
-------
Pattern Recognition and Machine Learning, by C. Bishop
"""

import os.path
import time
import numpy as np

import LearnAlgGMM as LA
from MLUtil import logsumexp

EPS = np.finfo(float).eps

class EMLearnerGMM( LA.LearnAlgGMM ):

  def __init__( self, gmm, savefilename='GMMtrace', nIter=100, \
                    initname='kmeans',  convTHR=1e-10, \
                    printEvery=5, saveEvery=5,\
                    **kwargs ):
    self.gmm = gmm
    self.savefilename = savefilename
    self.initname = initname
    self.convTHR = convTHR
    self.Niter = nIter
    self.printEvery = printEvery
    self.saveEvery = saveEvery
    self.SavedIters = dict()
    
  def init_params( self, X, **kwargs): 
    '''Initialize internal parameters w,mu,Sigma
          using specified "initname" and given data X
 
       Returns
       -------
        nothing. internal model params created (w,Mu,Sigma)
    '''
    self.gmm.D = X.shape[1]
    resp = self.init_resp( X, self.gmm.K, **kwargs )
    self.M_step( X, resp )
            
            
  def fit( self, X, seed=None):
    self.start_time = time.time()
    prevBound = -np.inf
    status = 'max iters reached.'

    for iterid in xrange(self.Niter):
      if iterid==0:
        self.init_params( X, seed=seed )
        resp, evBound = self.E_step( X )
      else:
        self.M_step( X, resp )
        resp, evBound = self.E_step( X )

      # Save and display progress
      self.save_state(iterid, evBound)
      self.print_state(iterid, evBound)

      # Check for Convergence!
      #  throw error if our bound calculation isn't working properly
      #    but only if the gap is greater than some tolerance
      #isValid = True
      isValid = prevBound < evBound or np.allclose( prevBound, evBound, rtol=self.convTHR )
      if not isValid:
        print '    prev = % .15e' % (prevBound)
        print '     cur = % .15e' % (evBound)
        raise Exception, 'evidence decreased!'
      if iterid >= self.saveEvery and np.abs(evBound-prevBound)/np.abs(evBound) <= self.convTHR:
        status = 'converged.'
        break
      prevBound = evBound

    #Finally, save, print and exit 
    self.save_state(iterid, evBound) 
    self.print_state(iterid, evBound, doFinal=True, status=status)


  def E_step( self, X):
    lpr = np.log( self.gmm.w ) + self.gmm.calc_soft_evidence_mat( X )
    lprPerItem = logsumexp( lpr, axis=1 )
    resp   = np.exp( lpr-lprPerItem[:,np.newaxis] ) 
    if not np.allclose( np.sum(resp,axis=1), 1.0 ):
      np.set_printoptions( linewidth=120, precision=3, suppress=True )      
      raise Exception, 'Responsibilities do not sum to one!'
    logEvidence = lprPerItem.sum()
    return resp, logEvidence
    
  def M_step( self, X, resp):
    '''M-step of the EM alg.
       Updates internal mixture model parameters
         to maximize the evidence of given data X  (aka probability of X)
       See Bishop PRML Ch.9 eqns 9.24, 9.25, and 9.26
         for updates to w, mu, and Sigma
    '''
    Nresp = resp.sum(axis=0)

    w = Nresp / Nresp.sum()

    wavg_X = np.dot(resp.T, X)
    mu = wavg_X / Nresp[:,np.newaxis]

    if self.gmm.covar_type == 'full':
      sigma = self.full_covar_M_step( X, resp, wavg_X, mu, Nresp )
      sigma += self.gmm.min_covar * np.eye( self.gmm.D )
    else:
      sigma = self.diag_covar_M_step( X, resp, wavg_X, mu, Nresp )
      sigma += self.gmm.min_covar

    mask = Nresp == 0
    if np.sum( mask ) > 0:
      w[ mask ] = 0    
      mu[ mask ] = 0
      sigma[ mask ] = 0

    self.gmm.w     = w
    self.gmm.mu    = mu
    self.gmm.Sigma = sigma
        
  def diag_covar_M_step( self, X, resp,  wavg_X, mu, Nresp ):
    wavg_X2 = np.dot(resp.T, X**2)
    wavg_M2 = mu**2 * Nresp[:,np.newaxis] 
    wavg_XM = wavg_X * mu
    sigma = (wavg_X2 -2*wavg_XM + wavg_M2 )
    return sigma / Nresp[:,np.newaxis]
    
  def full_covar_M_step( self, X, resp, wavg_X, mu, Nresp ):
    '''Update to full covariance matrix.  See Bishop PRML eq. 9.25
    '''
    N,D = X.shape
    Sigma = np.zeros( (self.gmm.K, D,D) ) 
    for k in range( self.gmm.K ):
      dX = X - mu[k]
      Sigma[k] = np.dot( dX.T, resp[:,k][:,np.newaxis]*dX ) / Nresp[k]
    return Sigma    
        
  def save_state( self, iterid, evBound ):
    if iterid==0: 
      mode = 'w' # create logfiles from scratch on initialization
    else:
      mode = 'a' # otherwise just append

    if iterid in self.SavedIters:
      return      
    self.SavedIters[iterid] = True

    if iterid % (self.saveEvery)==0:
      filename, ext = os.path.splitext( self.savefilename )
      with open( filename+'.iters', mode) as f:        
        f.write( '%d\n' % (iterid) )

      with open( filename+'.w', mode) as f:
        f.write( np2flatstr(self.gmm.w)+'\n')

      with open( filename+'.mu', mode) as f:
        f.write( np2flatstr(self.gmm.mu)+'\n' )
      
      with open( filename+'.sigma', mode) as f:
        f.write( np2flatstr(self.gmm.Sigma)+'\n' )
        
      with open( filename+'.evidence', mode) as f:
        f.write( '% .8e\n'% (evBound) )
        
def np2flatstr( X, fmt='% .6f' ):
  return ' '.join( [fmt % x for x in X.flatten() ] )  
        
#########################################################  Doc Tests
def verify_M_step():
  '''Doctest to verify that full and diag covariance give same M step result
    >>> import data.EasyToyGMMDataGenerator as Toy
    >>> import GMM
    >>> X = 10*np.random.randn( 100, 5) 
    >>> gdiag = GMM.GMM( K=3, covar_type='diag', D=5 )
    >>> gdiag.mu = Toy.Mu
    >>> gdiag.Sigma = Toy.Sigma
    >>> gdiag.w = Toy.w

    >>> gfull = GMM.GMM( K=3, covar_type='full', D=5 )
    >>> gfull.mu = Toy.Mu
    >>> gfull.Sigma = np.zeros( (3,5,5) )
    >>> gfull.w  = Toy.w
    >>> for k in range(3): gfull.Sigma[k] = np.diag( Toy.Sigma[k] )

    >>> em1 = EMLearnerGMM( gdiag )
    >>> resp, ev = em1.E_step( X )
    >>> em1.M_step( X, resp )
    
    >>> em2 = EMLearnerGMM( gfull )
    >>> resp2, ev = em2.E_step( X )
    >>> em2.M_step( X, resp2 )
    
    >>> print np.allclose( resp, resp2 )
    True
    >>> Sigma2 = np.vstack( [np.diag( em2.gmm.Sigma[k] )  for k in range(3)] )
    >>> print np.allclose( Sigma2, em1.gmm.Sigma )
    True
  '''
  pass
  
  
if __name__ == "__main__":
  import doctest
  doctest.testmod()
