'''
 Online EM Algorithm for learning simple 
   exponential family models
   
Author: Mike Hughes (mike@michaelchughes.com)
'''
import numpy as np
import time

import LearnAlg

class OnlineEMLearnAlg( LearnAlg.LearnAlg ):

  def __init__( self, expfamModel, **kwargs ):
    super(OnlineEMLearnAlg, self).__init__( **kwargs )
    self.expfamModel = expfamModel

  def init_params( self, Data, **kwargs):
    self.expfamModel.set_obs_dims( Data )
    LP = dict()
    LP['resp'] = self.init_resp( Data['X'], self.expfamModel.K, **kwargs )
    SS = self.expfamModel.get_global_suff_stats( Data, LP )
    self.expfamModel.update_global_params( SS )

  def fit( self, DataGenerator, seed, Dtest=None ):
    self.start_time = time.time()
    rho = 1
    for iterid, Dchunk in enumerate(DataGenerator):
      if iterid==0:
        self.init_params( Dchunk, seed=seed )
      else:
        self.expfamModel.update_global_params( SS, rho )

      LP = self.expfamModel.calc_local_params( Dchunk )
      SS = self.expfamModel.get_global_suff_stats( Dchunk, LP )

      if Dtest is None:
        evBound = self.expfamModel.calc_evidence( Dchunk, SS, LP )
      else:
      	tLP = self.expfamModel.calc_local_params( Dtest )
      	tSS = self.expfamModel.get_global_suff_stats( Dtest, tLP)
      	evBound = self.expfamModel.calc_evidence( Dtest, tSS, tLP )
      
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
      print self.expfamModel.obsModel.qobsDistr[0].mu  
      print self.expfamModel.obsModel.qobsDistr[1].mu  
      print self.expfamModel.obsModel.qobsDistr[2].mu
    pass

