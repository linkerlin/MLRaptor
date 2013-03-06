'''
'''
import numpy as np

from .GaussDistr import GaussDistr
from .GaussWishDistr import GaussWishDistr

LOGPI = np.log(np.pi)
LOGTWO = np.log(2.00)
LOGTWOPI = np.log( 2.0*np.pi )
EPS = 10*np.finfo(float).eps

class GaussianObsCompSet( object ):

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
    return 'Gaussian-Wishart'

  def set_obs_dims( self, Data):
    self.D = Data['X'].shape[1]
    if self.obsPrior is not None:
      self.obsPrior.set_dims( self.D )

  def update_obs_params_EM( self, SS, **kwargs):
    for k in xrange( self.K ):
      self.qobsDistr[k] = GaussDistr( SS['mean'][k], SS['covar'][k]+self.min_covar*np.eye(self.D) )
  
  def update_obs_params_EM_stochastic(self, SS, rho):
    for k in xrange( self.K ):
      freshDistr = GaussDistr( SS['mean'][k], SS['covar'][k]+self.min_covar )
      self.qobsDistr[k].mu = (1-rho)*self.qobsDistr[k].mu + rho*freshDistr.mu
      self.qobsDistr[k].Sigma = (1-rho)*self.qobsDistr[k].Sigma + rho*freshDistr.Sigma
             				 
  def update_obs_params_VB( self, SS, **kwargs):
    for k in xrange( self.K ):
      postDistr = self.obsPrior.getPosteriorDistr(SS['N'][k], SS['mean'][k],SS['covar'][k] )
      self.qobsDistr[k] = postDistr
      
  def update_obs_params_VB_stochastic( self, SS, rho, Ntotal):
    if Ntotal is None:
      ampF = 1
    else:
      ampF = Ntotal/SS['Nall']
    for k in xrange( self.K ):      
      postDistr = self.obsPrior.getPosteriorDistr( \
                       ampF*SS['N'][k], SS['mean'][k], SS['covar'][k] )
      self.qobsDistr[k] = postDistr
    
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
        
  def get_global_suff_stats( self, Data, SS, LP ):
    ''' Suff Stats
    '''
    if type(Data) is dict:
      X = Data['X']
    else:
      X = Data
    resp = LP['resp']

    SS['mean'] = np.dot( resp.T, X ) / SS['N'][:, np.newaxis]
    SS['covar'] = np.empty( (self.K, self.D, self.D) )
    for k in xrange( self.K):
      Xdiff = X - SS['mean'][k]
      SS['covar'][k] = np.dot( Xdiff.T * resp[:,k], Xdiff )
      SS['covar'][k] /= SS['N'][k]
    return SS
  
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
    lpX = -self.D*LOGTWOPI*np.ones( self.K )
    dist = np.zeros( self.K)
    for k in range( self.K ):
      lpX[k] += self.qobsDistr[k].ElogdetLam - self.D/self.qobsDistr[k].kappa \
                - self.qobsDistr[k].dF* self.qobsDistr[k].traceW( SS['covar'][k] )  \
                - self.qobsDistr[k].dF* self.qobsDistr[k].dist_mahalanobis( SS['mean'][k] )
    return 0.5*np.inner(SS['N'],lpX)
    
  def E_logpPhi( self ):
    return self.E_logpLam() + self.E_logpMu()
      
  def E_logqPhi( self ):
    return self.E_logqLam() + self.E_logqMu()
  
  def E_logpMu( self ):
    ''' First four RHS terms (inside sum over K) in Bishop 10.74
    '''
    lp = np.empty( self.K)    
    for k in range( self.K ):
      mWm = self.qobsDistr[k].dist_mahalanobis( self.obsPrior.m )
      lp[k] = self.qobsDistr[k].ElogdetLam \
                -self.D*self.obsPrior.kappa/self.qobsDistr[k].kappa \
                -self.obsPrior.kappa*self.qobsDistr[k].dF*mWm
    lp += self.D*( np.log( self.obsPrior.kappa ) - LOGTWOPI)
    return 0.5*lp.sum()
    
  def E_logpLam( self ):
    ''' Last three RHS terms in Bishop 10.74
    '''
    lp = np.empty( self.K) 
    for k in xrange( self.K ):
      lp[k] = 0.5*(self.obsPrior.dF - self.D -1)*self.qobsDistr[k].ElogdetLam
      lp[k] -= 0.5*self.qobsDistr[k].dF*self.qobsDistr[k].traceW(self.obsPrior.invW)
    return lp.sum() + self.K * self.obsPrior.logWishNormConst() 
    
  def E_logqMu( self ):
    ''' First two RHS terms in Bishop 10.77
    '''    
    lp = np.zeros( self.K)
    for k in xrange( self.K):
      lp[k] = 0.5*self.qobsDistr[k].ElogdetLam \
              + 0.5*self.D*( np.log( self.qobsDistr[k].kappa ) - LOGTWOPI )
    return lp.sum() - 0.5*self.D*self.K
                     
  def E_logqLam( self ):
    ''' Last two RHS terms in Bishop 10.77
    '''
    lp = np.zeros( self.K)
    for k in xrange( self.K):
      lp[k] -= self.qobsDistr[k].entropyWish()
    return lp.sum()
