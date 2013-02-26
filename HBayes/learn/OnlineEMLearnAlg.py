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

class OnlineEMLearnAlg( LearnAlg.LearnAlg ):

  def __init__( self, mixmodel, **kwargs ):
    super(OnlineEMLearnAlg, self).__init__( **kwargs )
    self.mixmodel = mixmodel

  def init_params( self, Data, **kwargs):
    self.mixmodel.set_dims( Data )
    LP = dict()
    LP['resp'] = self.init_resp( Data['X'], self.mixmodel.K, **kwargs )
    SS = self.mixmodel.get_global_suff_stats( Data, LP )
    self.mixmodel.update_global_params( SS )

  def fit( self, DataGenerator, seed, Dtest=None ):
    self.start_time = time.time()
    rho = 1
    for iterid, Dchunk in enumerate(DataGenerator):
      if iterid==0:
        self.init_params( Dchunk, seed=seed )
      else:
        self.mixmodel.update_global_params( SS, rho )

      LP = self.mixmodel.calc_local_params( Dchunk )
      SS = self.mixmodel.get_global_suff_stats( Dchunk, LP )

      if Dtest is None:
        evBound = self.mixmodel.calc_evidence( Dchunk, SS, LP )
      else:
      	tLP = self.mixmodel.calc_local_params( Dtest )
      	tSS = self.mixmodel.get_global_suff_stats( Dtest, tLP)
      	evBound = self.mixmodel.calc_evidence( Dtest, tSS, tLP )
      
      rho = ( iterid + self.rhodelay )**(-1*self.rhoexp)

      # Save and display progress
      self.save_state(iterid, evBound)
      self.print_state(iterid, evBound, rho=rho)

    status = 'all data gone.'
		#Finally, save, print and exit 
    self.save_state(iterid, evBound) 
    self.print_state(iterid, evBound, doFinal=True, status=status)


  def save_state( self, iterid, evBound ):
    np.set_printoptions( linewidth=120, precision=2, suppress=True)
    if iterid % 50 == 0:
      print self.mixmodel.obsDistr[0].mu  
      print self.mixmodel.obsDistr[1].mu  
      print self.mixmodel.obsDistr[2].mu
    pass

