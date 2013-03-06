'''
'''
import numpy as np

from .GaussianDistr2 import GaussianDistr2
from .GaussWishDistrIndep import GaussWishDistrIndep

LOGPI = np.log(np.pi)
LOGTWO = np.log(2.00)
LOGTWOPI = np.log( 2.0*np.pi )
EPS = 10*np.finfo(float).eps

class GaussObsCompSet2( object ):

  def __init__( self, K, qType='EM', obsPrior=None, min_covar=1e-8):
    self.K = K
    self.qType = qType
    self.obsPrior = obsPrior
    self.min_covar = min_covar
    self.qobsDistr = [None for k in xrange(K)]
    self.D = None

  def to_string(self):
    return 'Gaussian distribution'
  
  def to_string_prior(self):
    return 'Gaussian on \mu, Wishart on \Lam'

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

    SS['x']   = np.dot( resp.T, X )
    SS['xxT'] = np.empty( (self.K, self.D, self.D) )
    for k in xrange( self.K):
      SS['xxT'][k] = np.dot( X.T * resp[:,k], X )
    return SS

  ################################################################## Param updates

  def update_global_params( self, SS, rho=None, Ntotal=None):
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
      self.qobsDistr[k] = self.obsPrior.getPosteriorDistr( SS['N'][k], SS['x'][k],SS['xxT'][k] )

  def update_obs_params_VB_stochastic( self, SS, rho, Ntotal):
    pass   

  def update_obs_params_EM( self, SS, **kwargs):
    for k in xrange( self.K ):      
      mean    = SS['x'][k]/SS['N'][k]
      covMat  = SS['xxT'][k]/SS['N'][k] - np.outer(mean,mean)
      covMat  += self.min_covar *np.eye( self.D )      
      precMat = np.linalg.pinv( covMat )
      #precMat = np.linalg.solve( covMat, np.eye(self.D) )

      #if self.obsPrior is not None:
      #  precMat += self.obsPrior.LamPrior.invW
      #  mean += self.obsPrior.muPrior.m        
      self.qobsDistr[k] = GaussianDistr2( mean, precMat )
      
      
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
       Bishop PRML eq. 10.71
    '''
    lpX = np.zeros( self.K )
    for k in xrange( self.K ):
      LamD = self.qobsDistr[k].LamD
      muD  = self.qobsDistr[k].muD
      lpX[k]  = 0.5*SS['N'][k]*LamD.E_logdetLam()
      lpX[k] -= 0.5*SS['N'][k]*self.D*LOGTWOPI
      lpX[k] -= 0.5*SS['N'][k]*LamD.E_traceLambda( muD.invL )

      xmxmT  =  SS['xxT'][k] -2*np.outer(SS['x'][k],muD.m) + SS['N'][k]*np.outer(muD.m, muD.m)
      lpX[k] -= 0.5*LamD.E_traceLambda( xmxmT )
      assert lpX[k].size== 1
    return lpX.sum()
    
  def E_logpPhi( self ):
    return self.E_logpLam() + self.E_logpMu()
      
  def E_logqPhi( self ):
    return self.E_logqLam() + self.E_logqMu()
  
  def E_logpMu( self ):
    '''
    '''
    muP = self.obsPrior.muD
    lp = muP.get_log_norm_const() * np.ones( self.K )   
    for k in range( self.K ):
      muD = self.qobsDistr[k].muD
      lp[k] -= 0.5*np.trace( np.dot(muP.L, muD.invL) )
      lp[k] -= 0.5*muP.dist_mahalanobis( muD.m )
    return lp.sum()
    
  def E_logpLam( self ):
    '''
    '''
    LamP = self.obsPrior.LamD
    lp = LamP.get_log_norm_const() * np.ones( self.K )
    for k in xrange( self.K ):
      LamD = self.qobsDistr[k].LamD
      lp[k] += 0.5*( LamP.v - LamP.D - 1 )*LamD.E_logdetLam()
      lp[k] -= 0.5*LamD.E_traceLambda( LamP.invW )
    return lp.sum() 
    
  def E_logqMu( self ):
    ''' Return negative entropy!
    '''    
    lp = np.zeros( self.K)
    for k in xrange( self.K):
      lp[k] = self.qobsDistr[k].muD.get_entropy()
    return -1*lp.sum()
                     
  def E_logqLam( self ):
    ''' Return negative entropy!
    '''
    lp = np.zeros( self.K)
    for k in xrange( self.K):
      lp[k] = self.qobsDistr[k].LamD.get_entropy()
    return -1*lp.sum()
