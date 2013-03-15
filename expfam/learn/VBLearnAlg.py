'''
 Variational Bayes learning algorithm

Author: Mike Hughes (mike@michaelchughes.com)
'''
import numpy as np
import time

from .LearnAlg import LearnAlg

class VBLearnAlg( LearnAlg ):

  def __init__( self, expfamModel, **kwargs ):
    super(type(self), self).__init__( **kwargs )
    self.expfamModel = expfamModel

  def fit( self, Data, seed ):
    self.start_time = time.time()
    status = "max iters reached."
    prevBound = -np.inf
    evBound = -1
    for iterid in xrange(self.Niter):
      if iterid==0:
        self.init_global_params( Data, seed )        
      else:
        # M-step
        self.expfamModel.update_global_params( SS ) 
      
      # E-step 
      LP = self.expfamModel.calc_local_params( Data )

      SS = self.expfamModel.get_global_suff_stats( Data, LP )
      evBound = self.calc_evidence( Data, SS, LP )

      # Save and display progress
      self.save_state(iterid, evBound)
      self.print_state(iterid, evBound)

      # Check for Convergence!
      #  throw error if our bound calculation isn't working properly
      #    but only if the gap is greater than some tolerance
      isConverged = self.verify_evidence( evBound, prevBound )

      if isConverged:
        status = 'converged.'
        break
      prevBound = evBound

    #Finally, save, print and exit 
    self.save_state(iterid, evBound, doFinal=True) 
    self.print_state(iterid, evBound, doFinal=True, status=status)