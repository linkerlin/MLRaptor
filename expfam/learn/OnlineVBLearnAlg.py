'''
 Online/Stochastic Variational Bayes learning algorithm

Author: Mike Hughes (mike@michaelchughes.com)
'''
import numpy as np
import time

from .LearnAlg import LearnAlg

class OnlineVBLearnAlg( LearnAlg ):

  def __init__( self, expfamModel, **kwargs ):
    super(type(self),self).__init__( **kwargs )
    self.expfamModel = expfamModel
    self.Niter = '' # empty

  def fit( self, DataGenerator, seed, Ntotal=10000, Dtest=None ):
    self.start_time = time.time()
    rho = 1.0
    tLP =None
    for iterid, Dchunk in enumerate(DataGenerator):
      if iterid==0:
        self.init_global_params( Dchunk, seed )
      else:
        self.expfamModel.update_global_params( SS, rho, Ntotal=Ntotal )

      LP = self.expfamModel.calc_local_params( Dchunk )
      SS = self.expfamModel.get_global_suff_stats( Dchunk, LP )

      evBound = self.calc_evidence( Dchunk, SS, LP, Dtest )
      
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

