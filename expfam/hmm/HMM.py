'''
  HMM.py
     Bayesian parametric hidden markov model with a finite number of components K

  Provides code for performing variational Bayesian inference/plain vanilla EM
     
 Author: Mike Hughes (mike@michaelchughes.com)

 Parameters
 -------
    K        : # of components/states
    alpha0   : scalar hyperparam. of symmetric Dirichlet prior on Markov trans. matrix.

'''

import numpy as np
from scipy.special import gammaln, digamma
from ..util.MLUtil import logsumexp

EPS = 10*np.finfo(float).eps

class HMM( object ):

  def __init__( self, K=3, alpha0=1.0, qType='EM', **kwargs):
    self.qType = qType
    self.K = K
    self.alpha0 = alpha0
    self.PiInit = 1/K*np.ones( K )

  def get_info_string( self):
    return 'Finite Hidden Markov model with %d components' % (self.K)

  def to_string( self):
    if self.qType == 'VB':
      return np2flatstr( self.alpha )
    else:
      return np2flatstr( self.PiMat )
    	    	
  def calc_local_params( self, Data, LP):
    ''' E-step
          alternate between these updates until convergence
             q(Z)  (posterior on topic-token assignment)
         and q(W)  (posterior on group-topic distribution)
    '''
    if self.qType == 'EM':
      resp, respPair = HMMUtil.FwdBwdAlg( self.InitPi, self.PiMat, LP['E_log_soft_ev'] )
      LP['resp'] = resp
      LP['respPair'] = respPair
      return LP
    
    

  ###################################################################  Global Suff Stat
  def get_global_suff_stats( self, Data, SS, LP ):
    ''' 
    '''
    SS['N'] = np.sum( LP['resp'], axis=0 )
    SS['Ntotal'] = SS['N'].sum()
    SS['Ctrans'] = np.zeros( self.K, self.K )
    for ii in xrange( Data['nSeq'] ):
      for tt in xrange( Data['Tstart'][ii]+1, Data['Tstop'][ii] ):
        SS['Ctrans'] += LP['respPair'][tt]
    return SS
    
  ###################################################################  Global M-step
  def update_global_params( self, SS, rho=None, **kwargs ):
    if self.qType == 'EM':
      self.update_global_params_EM( SS )
    elif self.qType == 'VB':
      self.update_global_params_VB( SS )
    elif self.qType == 'oVB':
      if rho is None or rho==1 or Ntotal is None:
        self.update_global_params_VB( SS )
      else: 
        self.update_global_params_onlineVB( SS, rho, Ntotal )
      
  def update_global_params_EM( self, SS ):
    self.PiMat = self.alpha0 + SS['Ctrans']
    self.PiMat /= np.sum( self.PiMat, axis=1)[:,np.newaxis]
    
    
  ###################################################################  VB ELBO calc
  def calc_evidence( self, Data, SS, LP ):
    if self.qType == 'EM':
      return LP['evidence']
    return self.E_logpZ( LP ) - self.E_logqZ( LP ) \
           + self.E_logpPi( LP )   - self.E_logqPi(LP)
           
  def E_logpZ( self, GroupIDs, LP ):
    ElogpZ = 0
    for gg in xrange( len(GroupIDs) ):
      ElogpZ += np.sum( LP['resp'][GroupIDs[gg]] * LP['Elogw_perGroup'][gg] )
    return ElogpZ
    
  def E_logqZ( self, GroupIDs, LP ):
    ElogqZ = np.sum( LP['resp'] * np.log(EPS+LP['resp'] ) )
    return  ElogqZ    

  def E_logpW( self, LP ):
    nGroup = len(LP['alpha_perGroup'])
    ElogpW = gammaln(self.K*self.alpha0)-self.K*gammaln(self.alpha0)    
    ElogpW *= nGroup  # same prior over each group of data!
    for gg in xrange( nGroup ):
      ElogpW += (self.alpha0-1)*LP['Elogw_perGroup'][gg].sum()
    return ElogpW
 
  def E_logqW( self, LP ):
    ElogqW = 0
    for gg in xrange( len(LP['alpha_perGroup']) ):
      a_gg = LP['alpha_perGroup'][gg]
      ElogqW +=  gammaln(  a_gg.sum()) - gammaln(  a_gg ).sum() \
                  + np.inner(  a_gg -1,  LP['Elogw_perGroup'][gg] )
    return ElogqW
