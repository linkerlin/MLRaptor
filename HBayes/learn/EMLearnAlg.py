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

  def __init__( self, mixmodel, **kwargs ):
    super(EMLearnAlg, self).__init__( **kwargs )
    self.mixmodel = mixmodel

  def init_params( self, Data, **kwargs):
    self.mixmodel.set_dims( Data )
    LP = dict()
    LP['resp'] = self.init_resp( Data['X'], self.mixmodel.K, **kwargs )
    SS = self.mixmodel.get_global_suff_stats( Data, LP )
    self.mixmodel.update_global_params( SS )

  def fit( self, Data, seed ):
    self.start_time = time.time()
    status = 'max iters reached.'
    prevBound = -np.inf

    for iterid in xrange(self.Niter):
      if iterid==0:
        self.init_params( Data, seed=seed )
        LP = self.mixmodel.calc_local_params( Data )
      else:
        self.mixmodel.update_global_params( SS )  
        LP = self.mixmodel.calc_local_params( Data )

      SS = self.mixmodel.get_global_suff_stats( Data, LP )
      evBound = self.mixmodel.calc_evidence( Data, SS, LP )

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
    self.save_state(iterid, evBound) 
    self.print_state(iterid, evBound, doFinal=True, status=status)


  def save_state( self, iterid, evBound ):
    pass

