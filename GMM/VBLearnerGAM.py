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

class VBLearnerGAM( LA.LearnAlgGMM ):

  def __init__( self, qgmm, savefilename='GMMtrace', nIter=100, \
                    initname='kmeans',  convTHR=1e-10, \
                    printEvery=5, saveEvery=5, \
                    **kwargs ):
    self.qgmm = qgmm    
    self.savefilename = savefilename
    self.initname = initname
    self.convTHR = convTHR
    self.Niter = nIter
    self.printEvery = printEvery
    self.saveEvery = saveEvery
    self.SavedIters = dict()
    
  def init_params( self, Data, **kwargs ): 
    self.qgmm.D = Data['X'].shape[1]
    resp = self.init_resp( Data['X'], self.qgmm.K, **kwargs )
    SS = self.qgmm.calc_suff_stats( Data, resp)
    return self.qgmm.M_step( SS )
      
  def fit( self, Data, seed=None):
    self.start_time = time.time()
    prevBound = -np.inf
    status = 'max iters reached.'
    for iterid in xrange(self.Niter):
      if iterid==0:
        LP = self.init_params( Data, seed=seed )
      else:
        LP = self.qgmm.M_step( SS )
      resp = self.qgmm.E_step( Data, LP )
      SS = self.qgmm.calc_suff_stats( Data, resp )
      evBound = self.qgmm.calc_ELBO(  Data, resp, SS, LP )

      # Save and display progress
      self.save_state(iterid, evBound)
      self.print_state(iterid, evBound)

      # Check for Convergence!
      #  throw error if our bound calculation isn't working properly
      #    but only if the gap is greater than some tolerance
      
      isEqual = np.allclose( prevBound, evBound, atol=self.convTHR, rtol=self.convTHR )
      isValid = prevBound < evBound or isEqual
      if not isValid:
        print 'WARNING: evidence decreased!'
        print '    prev = % .15e' % (prevBound)
        print '     cur = % .15e' % (evBound)
        #raise Exception, 'evidence decreased!'
      if iterid >= 3 and isEqual:
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
      
    if iterid in self.SavedIters:
      return      
    self.SavedIters[iterid] = True

    if iterid % (self.saveEvery)==0:
      filename, ext = os.path.splitext( self.savefilename )

      for k in range( self.qgmm.K):
        with open( filename+'.qObsComp%03d'%(k), mode) as f:
          f.write( self.qgmm.to_obs_string(k) +'\n' )
      
      with open( filename+'.iters', mode) as f:
        f.write( '%d\n' % (iterid) )
        
      with open( filename+'.evidence', mode) as f:
        f.write( '% .8e\n'% (evBound) )
