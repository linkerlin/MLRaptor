'''
 Variational bayes learning algorithm

Author: Mike Hughes (mike@michaelchughes.com)
'''
import numpy as np
import time

import LearnAlg

class OnlineVBLearnAlg( LearnAlg.LearnAlg ):

  def __init__( self, qmixmodel, **kwargs ):
    super(OnlineVBLearnAlg, self).__init__( **kwargs )
    self.qmixmodel = qmixmodel

  def init_params( self, Data, **kwargs):
    self.qmixmodel.set_dims( Data )
    LP = dict()
    LP['resp'] = self.init_resp( Data['X'], self.qmixmodel.K, **kwargs )
    SS = self.qmixmodel.get_global_suff_stats( Data, LP )
    self.qmixmodel.update_global_params( SS )
    if 'GroupIDs' in Data:
      LP = self.init_params_perGroup( Data, LP)
    return LP

  def init_params_perGroup(self, Data, LP ):
    GroupIDs = Data['GroupIDs']
    LP['N_perGroup'] = np.zeros( (len(GroupIDs),self.qmixmodel.K)  )
    for gg in range( len(GroupIDs) ):
      LP['N_perGroup'][gg] = np.sum( LP['resp'][ GroupIDs[gg] ], axis=0 )
    return LP

  def fit( self, DataGenerator, seed, Ntotal=10000, Dtest=None ):
    self.start_time = time.time()
    rho = 1.0
    tLP =None
    for iterid, Dchunk in enumerate(DataGenerator):
      if iterid==0:
        LP = self.init_params( Dchunk, seed=seed )
        LP = self.qmixmodel.calc_local_params( Dchunk, LP )
      else:
        self.qmixmodel.update_global_params( SS, rho, Ntotal )
        LP = self.qmixmodel.calc_local_params( Dchunk, LP )
      SS = self.qmixmodel.get_global_suff_stats( Dchunk, LP )

      if Dtest is None:
        evBound = self.qmixmodel.calc_evidence( Dchunk, SS, LP )
      else:
      	tLP = self.qmixmodel.calc_local_params( Dtest, tLP )
      	tSS = self.qmixmodel.get_global_suff_stats( Dtest, tLP)
      	evBound = self.qmixmodel.calc_evidence( Dtest, tSS, tLP )
      
      rho = ( iterid + self.rhodelay )**(-1*self.rhoexp)

      # Save and display progress
      self.save_state(iterid, evBound)
      self.print_state(iterid, evBound, rho=rho)

    status = 'all data gone.'
    #Finally, save, print and exit 
    self.save_state(iterid, evBound) 
    self.print_state(iterid, evBound, doFinal=True, status=status)
    return LP

  def save_state( self, iterid, evBound ):
    np.set_printoptions( linewidth=120, precision=2, suppress=True)
    if iterid % 50 == 0:
      print self.qmixmodel.qobsDistr[0].m  
      print self.qmixmodel.qobsDistr[1].m  
      print self.qmixmodel.qobsDistr[2].m  
    pass

