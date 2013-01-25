"""
Online (streaming) version of Expectation Maximization
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
  em    = onlineEMLearnerGMM( mygmm, Niter=100 )
  em.fit( X )

Related
-------
  GMM.py
  EMLearnerGMM.py

References
-------
  Pattern Recognition and Machine Learning, by C. Bishop
"""
import numpy as np
import time
import argparse
import os.path
import sys

import LearnAlgGMM as LA
from MLUtil import logsumexp

class onlineEMLearnerGMM( LA.LearnAlgGMM ):
  

  def __init__( self, gmm, min_covar=0.01, savefilename='GMMtrace', \
                      initname='kmeans', Niter=10, printEvery=25, saveEvery=5, \
                      kappa=0.5, delay=1.0 ):
    self.gmm = gmm
    self.min_covar = min_covar
    self.kappa = float( kappa )
    self.delay = float( delay )
    self.saveEvery=saveEvery
    self.printEvery=printEvery
    self.savefilename = savefilename
    self.initname = initname
    self.needInitFlag = True

  def init_params( self, X, seed): 
    np.random.seed( seed )
    self.gmm.D = X.shape[1]
    resp = self.init_resp( X )
    self.M_step( X, resp, rho=1.0 )

  def fit( self, DataGenerator, seed=None ):
    self.start_time = time.time()
    prevBound = -np.inf
    status = 'max iters reached.'

    for iterid, Xchunk in enumerate(DataGenerator):
      if iterid==0:
        self.init_params( Xchunk, seed )
        evBound = self.fit_chunk( Xchunk, iterid )
      else:
        evBound = self.fit_chunk( Xchunk, iterid )

      # Save and display progress
      self.save_state(iterid, evBound)
      self.print_state(iterid, evBound)

    #Finally, save, print and exit 
    self.save_state(iterid, evBound) 
    self.print_state(iterid, evBound, doFinal=True, status=status)
    
  def fit_chunk( self, Xchunk, iterid ):
    rho = float( iterid + self.delay )**(-1*self.kappa)
    resp, evBound = self.E_step( Xchunk )
    self.M_step( Xchunk, resp, rho )
    return evBound
    
  def E_step( self, Xchunk ):
    lpr = np.log( self.gmm.w ) + self.gmm.calc_soft_evidence_mat( Xchunk )
    lprPerItem = logsumexp(lpr, axis=1)
    logEvidence = lprPerItem.sum()
    resp   = np.exp(lpr - lprPerItem[:, np.newaxis])
    return resp, logEvidence
  
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
    
    self.w = (1-rho)*self.w + rho*wChunk
    self.Mu = (1-rho)*self.Mu + rho*muChunk
    self.Sigma = (1-rho)*self.Sigma + rho*sigChunk
    
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
