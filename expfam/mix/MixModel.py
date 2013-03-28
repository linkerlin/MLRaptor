'''
  MixModel.py
     Bayesian parametric mixture model with a finite number of components K

  Provides code for performing variational Bayesian inference,
     with the EM (Expectation-Maximization) Algorithm as a special case.

 Author: Mike Hughes (mike@michaelchughes.com)

 Parameters
 -------
   K        : # of components
   alpha0   : scalar hyperparameter of symmetric Dirichlet prior on mix. weights

'''

import numpy as np
import scipy.io
from scipy.special import gammaln, digamma
from ..util.MLUtil import logsumexp, np2flatstr, flatstr2np

EPS = 10*np.finfo(float).eps

class MixModel( object ):

  def __init__( self, K=3, alpha0=1.0, qType='VB', **kwargs ):
    self.qType = qType
    self.K = K
    self.alpha0 = alpha0

  '''
  def save_params( self, fname, saveext):
    if saveext == 'txt':
      outpath = fname + 'AllocModel.txt'
      with open( outpath, 'a') as f:
        f.write( self.to_string() + '\n')
    elif saveext == 'mat':
      outpath = fname + 'AllocModel.mat'
      scipy.io.savemat( outpath, self.to_dict(), oned_as='row')
  '''
  
  def to_dict(self): 
    if self.qType.count('VB') >0:
      return dict( alpha=self.alpha)
    elif self.qType == 'EM':
      return dict( w=self.w )  

  def to_string(self):
    if self.qType == 'VB' or self.qType == 'oVB':
      return np2flatstr( self.alpha )
    elif self.qType == 'EM':
      return np2flatstr( self.w )
      
  def get_info_string( self):
    return 'Finite mixture model with %d components. Dir prior param %.2f' % (self.K, self.alpha0)

  def get_human_global_param_string(self):
    if self.qType == 'EM':
      return np2flatstr( self.w, '%3.2f' )
    else:
      return np2flatstr( np.exp(self.Elogw), '%3.2f' )

  ############################################################## LP/SS Updates   
  def calc_local_params( self, Data, LP ):
    ''' 
    '''
    if self.qType.count('VB') > 0:
      lpr = self.Elogw + LP['E_log_soft_ev']
    elif self.qType.count('EM') > 0:
      lpr = np.log(self.w) + LP['E_log_soft_ev']
    lprPerItem = logsumexp( lpr, axis=1 )
    resp   = np.exp( lpr-lprPerItem[:,np.newaxis] )
    LP['resp'] = resp
    if self.qType == 'EM':
        LP['evidence'] = lprPerItem.sum()
    return LP
           
  def get_global_suff_stats( self, Data, SS, LP ):
    ''' 
    '''
    SS['N'] = np.sum( LP['resp'], axis=0 )
    SS['Ntotal'] = SS['N'].sum()
    return SS

  ############################################################## Global Param Updates  
  def update_global_params( self, SS, rho=None, Ntotal=None, **kwargs ):
    '''
    '''
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
    self.w = self.alpha0 + SS['N']
    self.w /= self.w.sum()
  
  def update_global_params_VB( self, SS ):
    self.alpha = self.alpha0 + SS['N']
    self.Elogw      = digamma( self.alpha ) - digamma( self.alpha.sum() )

  def update_global_params_onlineVB( self, SS, rho, Ntotal):
    ampF = Ntotal/SS['Ntotal']
    alphNew = self.alpha0 + ampF*SS['N']
    self.alpha   = rho*alphNew + (1-rho)*self.alpha
    self.Elogw      = digamma( self.alpha ) - digamma( self.alpha.sum() )
    
  ############################################################## VB evidence calc
  def calc_evidence( self, Data, SS, LP):
    if self.qType == 'EM':
      return LP['evidence']
    elif self.qType.count('VB') >0:
      return self.E_logpZ( LP ) - self.E_logqZ( LP ) \
           + self.E_logpW()   - self.E_logqW()
           
  def E_logpZ( self, LP ):
    ''' Bishop PRML eq. 10.72
    '''
    return np.sum( LP['resp'] * self.Elogw )
    
  def E_logqZ( self, LP ):
    ''' Bishop PRML eq. 10.75
    '''
    return np.sum(  LP['resp'] *np.log( LP['resp']+EPS) )
    
  def E_logpW( self ):
    ''' Bishop PRML eq. 10.73
    '''
    return gammaln(self.K*self.alpha0)-self.K*gammaln(self.alpha0) \
             + (self.alpha0-1)*self.Elogw.sum()
 
  def E_logqW( self ):
    ''' Bishop PRML eq. 10.76
    '''
    return gammaln(self.alpha.sum())-gammaln(self.alpha).sum() \
             + np.inner( (self.alpha-1), self.Elogw )
