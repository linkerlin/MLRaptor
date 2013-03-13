'''
 EM Algorithm for learning mixture models 

  Simply defines some generic initialization routines, based on 
     assigning cluster responsibilities (either hard or soft) to 
     cluster centers either learned from data (kmeans)
                         or selected at random from the data

Author: Mike Hughes (mike@michaelchughes.com)
'''
import numpy as np
import time

import LearnAlg

class EMLearnAlg( LearnAlg.LearnAlg ):

  def __init__( self, expfamModel, **kwargs ):
    super(EMLearnAlg, self).__init__( **kwargs )
    self.expfamModel = expfamModel

  def init_params( self, Data, **kwargs):
    self.expfamModel.set_obs_dims( Data )
    LP = dict()
    LP['resp'] = self.init_resp( Data['X'], self.expfamModel.K, **kwargs )
    SS = self.expfamModel.get_global_suff_stats( Data, LP )
    self.expfamModel.update_global_params( SS )

  def fit( self, Data, seed ):
    self.start_time = time.time()
    status = 'max iters reached.'
    prevBound = -np.inf

    for iterid in xrange(self.Niter):
      if iterid==0:
        self.init_params( Data, seed=seed )
        LP = self.expfamModel.calc_local_params( Data )
      else:
        self.expfamModel.update_global_params( SS )  
        LP = self.expfamModel.calc_local_params( Data )

      SS = self.expfamModel.get_global_suff_stats( Data, LP )
      evBound = self.expfamModel.calc_evidence( Data, SS, LP )

      # Save and display progress
      self.save_state(iterid, evBound)
      self.print_state(iterid, evBound)

      # Check for Convergence!
      #  throw error if our bound calculation isn't working properly
      #    but only if the gap is greater than some tolerance
      isConverged = self.verify_evidence( evBound, prevBound )

      if iterid >= self.saveEvery and isConverged:
        status = 'converged.'
        break
      prevBound = evBound

    #Finally, save, print and exit 
    self.save_state(iterid, evBound, True) 
    self.print_state(iterid, evBound, doFinal=True, status=status)

