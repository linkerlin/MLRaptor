'''
 Variational bayes learning algorithm

Author: Mike Hughes (mike@michaelchughes.com)
'''
import numpy as np
import time

from .LearnAlg import LearnAlg

class OnlineVBLearnAlg( LearnAlg ):

  def __init__( self, expfamModel, **kwargs ):
    super(type(self),self).__init__( **kwargs )
    self.expfamModel = expfamModel

  def init_params( self, Data, **kwargs):
    self.expfamModel.set_obs_dims( Data )
    LP = dict()
    LP['resp'] = self.init_resp( Data['X'], self.expfamModel.K, **kwargs )
    SS = self.expfamModel.get_global_suff_stats( Data, LP )
    self.expfamModel.update_global_params( SS, **kwargs )
    if 'GroupIDs' in Data:
      LP = self.init_params_perGroup( Data, LP)
    return LP

  def init_params_perGroup(self, Data, LP ):
    GroupIDs = Data['GroupIDs']
    LP['N_perGroup'] = np.zeros( (len(GroupIDs),self.expfamModel.K)  )
    for gg in range( len(GroupIDs) ):
      LP['N_perGroup'][gg] = np.sum( LP['resp'][ GroupIDs[gg] ], axis=0 )
    return LP

  def fit( self, DataGenerator, seed, Ntotal=10000, Dtest=None ):
    self.start_time = time.time()
    rho = 1.0
    tLP =None
    for iterid, Dchunk in enumerate(DataGenerator):
      if iterid==0:
        LP = self.init_params( Dchunk, Ntotal=Ntotal, rho=1.0, seed=seed )
      else:
        self.expfamModel.update_global_params( SS, rho, Ntotal=Ntotal )

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
    try:
      self.save_state(iterid, evBound, doFinal=True) 
      self.print_state(iterid, evBound, doFinal=True, status=status)
      return LP
    except UnboundLocalError:
      print 'No iterations performed.  Perhaps provided DataGen empty. Rebuild DataGen and try again.'


