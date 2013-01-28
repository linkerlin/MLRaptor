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

def np2flatstr( X, fmt='% .6f' ):
  return ' '.join( [fmt % x for x in X.flatten() ] )  
  
class OnlineEMLearnerGMM( LA.LearnAlgGMM ):
  
  def __init__( self,  gmm, savefilename='GMMtrace', \
                    initname='kmeans',  convTHR=1e-10, \
                    min_covar=0.01, printEvery=5, saveEvery=5, \
                    rhoexp=0.5, rhodelay=1.0, **kwargs ):
    self.gmm = gmm
    self.min_covar = min_covar
    self.rhoexp   = float( rhoexp )
    self.rhodelay = float( rhodelay )
    self.saveEvery=saveEvery
    self.printEvery=printEvery
    self.savefilename = savefilename
    self.initname = initname
    self.Niter = ''  # needed to extend LearnAlgGMM
    
  def init_params( self, X, **kwargs): 
    '''Initialize internal parameters w,mu,Sigma
          using specified "initname" and given data X
 
       Returns
       -------
        nothing. internal model params created (w,Mu,Sigma)
    '''
    self.gmm.D = X.shape[1]
    resp = self.init_resp( X, self.gmm.K, **kwargs )
    w,mu,Sigma = self.M_step( X, resp )
    self.gmm.w = w
    self.gmm.mu = mu
    self.gmm.Sigma = Sigma

  def fit( self, DataGenerator, seed=None ):
    self.start_time = time.time()
    prevBound = -np.inf

    for iterid, Xchunk in enumerate(DataGenerator):
      if iterid==0:
        self.init_params( Xchunk, seed=seed )
        evBound = self.fit_chunk( Xchunk, iterid )
      else:
        evBound = self.fit_chunk( Xchunk, iterid )

      # Save and display progress
      self.save_state(iterid, evBound)
      self.print_state(iterid, evBound)

    #Finally, save, print and exit 
    self.save_state(iterid, evBound) 
    self.print_state(iterid, evBound, doFinal=True, status='all data gone.')
    
  def fit_chunk( self, Xchunk, iterid ):
    rho = ( iterid + self.rhodelay )**(-1*self.rhoexp)
    resp, evBound = self.E_step( Xchunk )
    wChunk,muChunk,sigChunk = self.M_step( Xchunk, resp )
    self.gmm.w = (1-rho)*self.gmm.w + rho*wChunk
    self.gmm.mu = (1-rho)*self.gmm.mu + rho*muChunk
    self.gmm.Sigma = (1-rho)*self.gmm.Sigma + rho*sigChunk
    return evBound
    
  def E_step( self, Xchunk ):
    '''Expectation step

       Returns
       -------
          resp : NxK matrix, resp[n,k] = Pr(Z[n]=k | X[n],mu[k],Sigma[k])
    '''
    lpr = np.log( self.gmm.w ) + self.gmm.calc_soft_evidence_mat( Xchunk )
    lprPerItem = logsumexp(lpr, axis=1)
    logEvidence = lprPerItem.sum()
    resp   = np.exp(lpr - lprPerItem[:, np.newaxis])
    return resp, logEvidence
  
    
  def M_step( self, Xchunk, resp):
    '''M-step of the EM alg. on current chunk of data X
       See Bishop PRML Ch.9 eqns 9.24, 9.25, and 9.26
         for updates to w, mu, and Sigma
    '''
    Nresp = resp.sum(axis=0)

    w = Nresp / Nresp.sum()

    wavg_X = np.dot(resp.T, Xchunk)
    mu = wavg_X / Nresp[:,np.newaxis]

    if self.gmm.covar_type == 'full':
      sigma = self.full_covar_M_step( Xchunk, resp, wavg_X, mu, Nresp )
      sigma += self.gmm.min_covar*np.eye(self.gmm.D) 
    else:
      sigma = self.diag_covar_M_step( Xchunk, resp, wavg_X, mu, Nresp )
      sigma += self.gmm.min_covar

    mask = ( Nresp == 0 )
    if mask.sum() > 0:
      w[ mask ] = 0    
      mu[ mask ] = 0
      sigma[ mask ] = 0
      
    return w, mu, sigma
        
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
