'''
 Variational bayes learning algorithm

Author: Mike Hughes (mike@michaelchughes.com)
'''
import numpy as np
import time

import LearnAlg

class VBLearnAlg( LearnAlg.LearnAlg ):

  def __init__( self, qmixmodel, **kwargs ):
    super(VBLearnAlg, self).__init__( **kwargs )
    self.qmixmodel = qmixmodel

  def init_params( self, Data, **kwargs):
    self.qmixmodel.set_dims( Data )
    LP = dict()
    LP['resp'] = self.init_resp( Data['X'], self.qmixmodel.K, **kwargs )
    SS = self.qmixmodel.get_global_suff_stats( Data, LP )
    self.qmixmodel.update_global_params( SS )
    return SS

  def fit( self, Data, seed ):
    self.start_time = time.time()
    status = 'max iters reached.'
    prevBound = -np.inf

    for iterid in xrange(self.Niter):
      if iterid==0:
        SS = self.init_params( Data, seed=seed )
        LP = self.qmixmodel.calc_local_params( Data )
      else:
        # M-step
        SS = self.qmixmodel.get_global_suff_stats( Data, LP )
        self.qmixmodel.update_global_params( SS ) 
        # E-step 
        LP = self.qmixmodel.calc_local_params( Data )

      evBound = self.qmixmodel.calc_evidence( Data, SS, LP )

      # Save and display progress
      self.save_state(iterid, evBound)
      self.print_state(iterid, evBound)

      # Check for Convergence!
      #  throw error if our bound calculation isn't working properly
      #    but only if the gap is greater than some tolerance
      isValid = prevBound < evBound or np.allclose( prevBound, evBound, rtol=self.convTHR )
      self.verify_evidence( isValid )

      if iterid >= self.saveEvery and np.abs(evBound-prevBound)/np.abs(evBound) <= self.convTHR:
        status = 'converged.'
        break
      prevBound = evBound

    #Finally, save, print and exit 
    self.save_state(iterid, evBound) 
    self.print_state(iterid, evBound, doFinal=True, status=status)


  def save_state( self, iterid, evBound ):
    pass

