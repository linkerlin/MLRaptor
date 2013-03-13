'''
'''
import numpy as np

from .BernoulliDistr import BernoulliDistr
from .BetaDistr import BetaDistr

EPS = 10*np.finfo(float).eps

class BernObsCompSet( object ):

  def __init__( self, K, qType='EM', obsPrior=None, **kwargs):
    self.K = K
    self.qType = qType
    self.obsPrior = obsPrior
    self.qobsDistr = [None for k in xrange(K)]
    self.D = None

  def to_string(self):
    return 'Bernoulli distribution'
  
  def to_string_prior(self):
    if self.obsPrior is None:
      return 'None'
    return 'Beta distr'

  def set_obs_dims( self, Data):
    self.D = Data['X'].shape[1]
    if self.obsPrior is not None:
      self.obsPrior.set_dims( self.D )

  ################################################################## Suff stats
  def get_global_suff_stats( self, Data, SS, LP ):
    ''' Suff Stats
    '''
    if type(Data) is dict:
      X = Data['X']
    else:
      X = Data
    resp = LP['resp']

    SS['count']   = np.dot( resp.T, X )
    return SS

  ################################################################## Param updates

  def update_global_params( self, SS, rho=None, Ntotal=None, **kwargs):
    ''' M-step update
    '''
    if self.qType == 'EM':
      if rho is None:
        self.update_obs_params_EM( SS)
      else:
        self.update_obs_params_EM_stochastic( SS, rho )
    elif self.qType == 'VB':
      if rho is None:
        self.update_obs_params_VB( SS )
      else:
        self.update_obs_params_VB_stochastic( SS, rho, Ntotal )

  def update_obs_params_VB( self, SS, **kwargs):
    for k in xrange( self.K ):      
      self.qobsDistr[k] = self.obsPrior.getPosteriorDistr( SS['N'][k], SS['count'][k] )

  def update_obs_params_VB_stochastic( self, SS, rho, Ntotal, **kwargs):
    if Ntotal is None:
      ampF = 1
    else:
      ampF = Ntotal/SS['Ntotal']
    for k in xrange( self.K ):
      postDistr = self.obsPrior.getPosteriorDistr( ampF*SS['N'][k], ampF*SS['count'][k] )
      if self.qobsDistr[k] is None:
        self.qobsDistr[k] = postDistr
      else:
        self.qobsDistr[k].rho_update( rho, postDistr )

  def update_obs_params_EM( self, SS, **kwargs):
    for k in xrange( self.K ):      
      self.qobsDistr[k] = BernoulliDistr( SS['count'][k]/SS['N'][k] )
      #if self.obsPrior is not None:
      #  precMat += self.obsPrior.LamPrior.invW
      #  mean += self.obsPrior.muPrior.m        
      
      
  #########################################################  Soft Evidence Fcns  
  def calc_local_params( self, Data, LP):
    if self.qType == 'EM':
      LP['log_soft_ev'] = self.log_soft_ev_mat( Data['X'] )
    else:
      LP['E_log_soft_ev'] = self.E_log_soft_ev_mat( Data['X'] )
    return LP

  def log_soft_ev_mat( self, X ):
    ''' E-step update,  for EM-type
    '''
    N,D = X.shape
    lpr = np.empty( (X.shape[0], self.K) )
    for k in xrange( self.K ):
      lpr[:,k] = self.qobsDistr[k].log_pdf( X )
    return lpr 
      
  def E_log_soft_ev_mat( self, X ):
    ''' E-step update, for VB-type
    '''
    N,D = X.shape
    lpr = np.empty( (X.shape[0], self.K) )
    for k in xrange( self.K ):
      lpr[:,k] = self.qobsDistr[k].E_log_pdf( X )
    return lpr
  
  #########################################################  Evidence Bound Fcns  
  def calc_evidence( self, Data, SS, LP):
    if self.qType == 'EM': return 0 # handled by alloc model
    return self.E_logpX( LP, SS) \
           + self.E_logpPhi() - self.E_logqPhi()
  
  def E_logpX( self, LP, SS ):
    ''' E_{q(Z), q(Phi)} [ log p(X) ]
    '''
    lpX = np.zeros( self.K )
    for k in xrange( self.K ):
      cON = SS['count'][k]
      cOFF = SS['N'][k] - cON
      lpX[k] = np.sum( cON * self.qobsDistr[k].Elogphi )
      lpX[k] += np.sum(cOFF*self.qobsDistr[k].Elog1mphi )
    return lpX.sum()
    
  def E_logpPhi( self ):
    lp = self.obsPrior.get_log_norm_const()*np.ones( self.K)
    for k in xrange( self.K):
      lp[k] += np.sum( (self.obsPrior.a - 1)*self.qobsDistr[k].Elogphi )
      lp[k] += np.sum( (self.obsPrior.b - 1)*self.qobsDistr[k].Elog1mphi )
    return lp.sum()
          
  def E_logqPhi( self ):
    ''' Return negative entropy!
    '''    
    lp = np.zeros( self.K)
    for k in xrange( self.K):
      lp[k] = self.qobsDistr[k].get_entropy()
    return -1*lp.sum()
