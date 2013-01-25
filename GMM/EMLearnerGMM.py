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

def np2flatstr( X, fmt='% .6f' ):
  return ' '.join( [fmt % x for x in X.flatten() ] )  

class EMLearnerGMM( LA.LearnAlgGMM ):

  def __init__( self, gmm, min_covar=0.01, savefilename='GMMtrace', \
                      initname='kmeans', Niter=10, printEvery=25, saveEvery=5 ):
    self.gmm = gmm
    self.min_covar = min_covar
    self.savefilename = savefilename
    self.initname = initname
    self.Niter = Niter
    self.printEvery = printEvery
    self.saveEvery = saveEvery
    
  def init_params( self, X, seed): 
    np.random.seed( seed )
    self.gmm.D = X.shape[1]
    resp = self.init_resp( X, self.gmm.K )
    self.M_step( X, resp )
            
  def fit( self, X, seed=None, convTHR=1e-4):
    self.start_time = time.time()
    prevBound = -np.inf
    status = 'max iters reached.'

    for iterid in xrange(self.Niter):
      if iterid==0:
        self.init_params( X, seed )
        evBound = self.gmm.calc_evidence( X )
      else:
        resp, evBound = self.E_step( X )
        self.M_step( X, resp )

      # Check for Convergence!
      assert prevBound <= evBound
      if iterid > 3 and np.abs( evBound-prevBound )/np.abs(evBound) <= convTHR:
        status = 'converged.'
        break
      prevBound = evBound

      # Save and display progress
      self.save_state(iterid, evBound)
      self.print_state(iterid, evBound)

    #Finally, save, print and exit 
    self.save_state(iterid, evBound) 
    self.print_state(iterid, evBound, doFinal=True, status=status)
  
  def E_step( self, X):
    lpr = np.log( self.gmm.w ) + self.gmm.calc_soft_evidence_mat( X )
    lprPerItem = logsumexp(lpr, axis=1)
    logEvidence = lprPerItem.sum()
    resp   = np.exp(lpr - lprPerItem[:, np.newaxis])
    return resp, logEvidence
    
  def M_step( self, X, resp):
    '''M-step of the EM alg.
       Updates internal mixture model parameters
         to maximize the evidence of given data X  (aka probability of X)
       See Bishop PRML Ch.9 eqns 9.24, 9.25, and 9.26
         for updates to w, mu, and Sigma
    '''
    Nresp = resp.sum(axis=0)
    w = Nresp / ( Nresp.sum() + EPS )

    wavg_X = np.dot(resp.T, X)
    mu = wavg_X / (Nresp[:,np.newaxis] + EPS)

    wavg_X2 = np.dot(resp.T, X**2)
    wavg_M2 = mu**2 * Nresp[:,np.newaxis] 
    wavg_XM = wavg_X * mu
    sigma = wavg_X2 -2*wavg_XM + wavg_M2
    sigma /= (Nresp[:,np.newaxis] + EPS)
    
    self.gmm.w     = w
    self.gmm.mu    = mu
    self.gmm.Sigma = sigma + self.min_covar
        
  def save_state( self, iterid, evBound ):
    if iterid==0: 
      mode = 'w' # create logfiles from scratch on initialization
    else:
      mode = 'a' # otherwise just append
      
    if iterid % (self.saveEvery)==0:
      filename, ext = os.path.splitext( self.savefilename )
      with open( filename+'.w', mode) as f:
        f.write( np2flatstr(self.gmm.w)+'\n')

      with open( filename+'.mu', mode) as f:
        f.write( np2flatstr(self.gmm.mu)+'\n' )
      
      with open( filename+'.sigma', mode) as f:
        f.write( np2flatstr(self.gmm.Sigma)+'\n' )
        
      with open( filename+'.iters', mode) as f:
        f.write( '%d\n' % (iterid) )
        
      with open( filename+'.evidence', mode) as f:
        f.write( '% .6e\n'% (evBound) )
