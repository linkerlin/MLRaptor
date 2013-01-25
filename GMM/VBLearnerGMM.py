"""
Variational Bayesian learning algorithm
  for a Gaussian Mixture Model (GMM)

Author: Mike Hughes (mike@michaelchughes.com)

Supports:
  * multiple initializations
  * easy logging of parameter values/ ELBO traces across iterations
     (saved to files in results/ directory)

Usage
-------
To fit a 5-component GMM to data matrix X (each row is an 2D obs. vector)
  gw   = GaussWishDistr( dF=3, invW=np.eye(2) )
  qgmm = QGMM( alpha0=1.0, K=5, obsPrior=gw )
  vb   = VBLearnerGMM( qgmm, Niter=100 )
  vb.fit( X )

References
-------
Pattern Recognition and Machine Learning, by C. Bishop
"""
import os.path
import time
import numpy as np
import scipy.linalg

import LearnAlgGMM as LA
from MLUtil import logsumexp

EPS = np.finfo(float).eps

class VBLearnerGMM( LA.LearnAlgGMM ):

  def __init__( self, qgmm, savefilename='GMMtrace', \
                      initname='kmeans', Niter=10, printEvery=25, saveEvery=5 ):
    self.qgmm = qgmm    
    self.savefilename = savefilename
    self.initname = initname
    self.Niter = Niter
    self.printEvery = printEvery
    self.saveEvery = saveEvery
    
  def init_params( self, X, seed=None): 
    np.random.seed( seed )
    self.qgmm.D = X.shape[1]
    resp = self.init_resp( X )
    SS = self.qgmm.calc_suff_stats( X, resp)
    self.qgmm.M_step( SS )
      
  def fit( self, X, seed=None, convTHR=1e-10):
    self.start_time = time.time()
    prevBound = -np.inf
    status = 'max iters reached.'
    for iterid in xrange(self.Niter):
      if iterid==0:
        self.init_params( X, seed )
      else:
        self.qgmm.M_step( SS )
      resp = self.qgmm.E_step( X )
      SS = self.qgmm.calc_suff_stats( X, resp )
      evBound = self.qgmm.calc_ELBO( resp, SS )
      self.save_state(iterid, evBound)
      self.print_state(iterid, evBound)
      assert prevBound <= evBound
      if iterid > 3 and np.abs( evBound-prevBound )/np.abs(evBound) <= convTHR:
        status = 'converged.'
        break
      prevBound = evBound
    #Finally, save, print and exit 
    self.save_state(iterid, evBound) 
    self.print_state(iterid, evBound, doFinal=True, status=status)


  def save_state( self, iterid, evBound ):
    if iterid==0: 
      mode = 'w' # create logfiles from scratch on initialization
    else:
      mode = 'a' # otherwise just append
      
    if iterid % (self.saveEvery)==0:
      filename, ext = os.path.splitext( self.savefilename )
      with open( filename+'.alpha', mode) as f:
        f.write( self.qgmm.to_alpha_string() +'\n')

      for k in range( self.qgmm.K):
        with open( filename+'.qObsComp%03d'%(k), mode) as f:
          f.write( self.qgmm.to_obs_string(k) +'\n' )
      
      with open( filename+'.iters', mode) as f:
        f.write( '%d\n' % (iterid) )
        
      with open( filename+'.evidence', mode) as f:
        f.write( '% .6e\n'% (evBound) )
