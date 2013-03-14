'''
'''
import numpy as np

from .MultinomialDistr import MultinomialDistr
from .DirichletDistr import DirichletDistr

EPS = 10*np.finfo(float).eps

class MultObsCompSet( object ):

  def __init__( self, K, qType='VB', obsPrior=None, **kwargs):
    self.K = K
    self.qType = qType
    self.obsPrior = obsPrior
    self.qobsDistr = [None for k in xrange(K)]
    self.D = None

  def get_info_string(self):
    return 'Multinomial distribution'
  
  def get_info_string_prior(self):
    if self.obsPrior is None:
      return 'None'
    else:
      return 'Dirichlet'

  def set_obs_dims( self, Data):
    try:
      self.D = Data['nVocab']
    except KeyError:
      self.D = Data['X'].shape[1]
    if self.obsPrior is not None:
      self.obsPrior.set_dims( self.D )

  def save_params( self, filename):
    pass

  ################################################################## Suff stats
  def get_global_suff_stats( self, Data, SS, LP ):
    ''' Suff Stats
    '''
    resp = LP['resp']
    try:      
      SS['TermCount']   = np.zeros( (self.K, self.D) )      
      for docDict in Data['BoW']:
        self.increment_termcount_from_dict( docDict, SS['TermCount'], resp )
    except KeyError:      
      SS['TermCount']   = np.dot( resp.T, Data['X'] )
    return SS

  def increment_termcount_from_dict( self, CDict, TermCountMat, resp ):
    tokenID = 0
    for (termID,count) in CDict.items():
      TermCountMat[:,termID] += resp[tokenID] * count
      tokenID += 1

  ################################################################## Param updates

  def update_global_params( self, SS, rho=None, Ntotal=None, **kwargs):
    ''' M-step update
    '''
    if self.qType == 'EM':
        self.update_obs_params_EM( SS)

    elif self.qType.count('VB')>0:
      if rho is None:
        self.update_obs_params_VB( SS )
      else:
        self.update_obs_params_VB_stochastic( SS, rho, Ntotal )

  def update_obs_params_EM( self, SS, **kwargs):
    for k in xrange( self.K ):      
      self.qobsDistr[k] = MultinomialDistr( SS['TermCount'][k]/SS['N'][k] )

  def update_obs_params_VB( self, SS, **kwargs):
    for k in xrange( self.K):
      self.qobsDistr[k] = self.obsPrior.getPosteriorDistr( SS['TermCount'][k]/SS['N'][k] )

  def update_obs_params_VB_stochastic( self, SS, rho, Ntotal, **kwargs):
    pass
      
  #########################################################  Soft Evidence Fcns  
  def calc_local_params( self, Data, LP):
    if self.qType == 'EM':
      LP['E_log_soft_ev'] = self.log_soft_ev_mat( Data )
    else:
      LP['E_log_soft_ev'] = self.E_log_soft_ev_mat( Data )
    return LP

  def log_soft_ev_mat( self, Data ):
    ''' E-step update,  for EM-type
    '''
    N,D = X.shape
    lpr = np.empty( (Data['nObs'], self.K) )
    for k in xrange( self.K ):
      lpr[:,k] = self.qobsDistr[k].log_pdf( Data )
    return lpr 
      
  def E_log_soft_ev_mat( self, X ):
    ''' E-step update, for VB-type
    '''
    N,D = X.shape
    lpr = np.empty( (Data['nObs'], self.K) )
    for k in xrange( self.K ):
      lpr[:,k] = self.qobsDistr[k].E_log_pdf( Data )
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
      lpX[k] = np.sum( SS['TermCount'][k] * self.qobsDistr[k].Elogphi )
    return lpX.sum()
    
  def E_logpPhi( self ):
    lp = self.obsPrior.get_log_norm_const()*np.ones( self.K)
    for k in xrange( self.K):
      lp[k] += np.sum( (self.obsPrior.lamvec - 1)*self.qobsDistr[k].Elogphi )
    return lp.sum()
          
  def E_logqPhi( self ):
    ''' Return negative entropy!
    '''    
    lp = np.zeros( self.K)
    for k in xrange( self.K):
      lp[k] = self.qobsDistr[k].get_entropy()
    return -1*lp.sum()
