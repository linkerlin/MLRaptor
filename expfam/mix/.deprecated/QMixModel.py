'''
  Represents mean-field factorization of a 
    Bayesian mixture model with a finite number of components K

 Author: Mike Hughes (mike@michaelchughes.com)

 Parameters
 -------
   alpha0   : scalar hyperparameter of symmetric Dirichlet prior on mix. weights
   
 References
 -------
   Pattern Recognition and Machine Learning, by C. Bishop.
'''

import numpy as np
from scipy.special import gammaln, digamma
from ..util.MLUtil import logsumexp, np2flatstr, flatstr2np

EPS = 10*np.finfo(float).eps

class QMixModel( object ):

  def __init__( self, K, alpha0, **kwargs ):
    self.qType = 'VB'
    self.K = K
    self.alpha0 = alpha0

  def from_string(self, alphastr):
    self.alpha = flatstr2np( alphastr )

  def to_string(self):
    return np2flatstr( self.alpha )
    
  def get_info_string( self):
    return 'Finite mixture model with %d components' % (self.K)
    
  def calc_local_params( self, Data, LP ):
    ''' 
    '''
    lpr = self.Elogw + LP['E_log_soft_ev']
    lprPerItem = logsumexp( lpr, axis=1 )
    resp   = np.exp( lpr-lprPerItem[:,np.newaxis] )
    LP['resp'] = resp
    return LP
           
  def get_global_suff_stats( self, Data, SS, LP ):
    ''' 
    '''
    SS['N'] = np.sum( LP['resp'], axis=0 )
    SS['Nall'] = SS['N'].sum()
    return SS
    
  def update_global_params( self, SS, rho=None, Ntotal=None, **kwargs ):
    '''
    '''
    if Ntotal is None:
      ampF = 1
    else:
      ampF = Ntotal/SS['Nall']
    alphNew = self.alpha0 + ampF*SS['N']
    if rho is None or rho==1:
      self.alpha   = alphNew
    else:
      self.alpha   = rho*alphNew + (1-rho)*self.alpha
    self.Elogw      = digamma( self.alpha ) - digamma( self.alpha.sum() )
    
  def calc_evidence( self, Data, SS, LP):
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
