'''
  Represents mean-field factorization of a 
    Bayesian mixture model with a finite number of components K

 Author: Mike Hughes (mike@michaelchughes.com)

 Parameters
 -------
   alpha0   : scalar hyperparameter of symmetric Dirichlet prior on mix. weights
   
 Usage
 -------
   This class is abstract.  See QGMM.py.

 Inference
 -------
   See VBLearner.py  or EMLearner.py

 References
 -------
   Pattern Recognition and Machine Learning, by C. Bishop.
'''

import numpy as np
from scipy.special import gammaln, digamma
from ..util.MLUtil import logsumexp

EPS = 10*np.finfo(float).eps

class QMixModel( object ):

  def __init__( self, K, alpha0 ):
    self.K = K
    self.alpha0 = alpha0

  def calc_local_params( self, Data, LP=None ):
    ''' 
    '''
    LP = dict()
    lpr = self.Elogw + self.E_log_soft_ev_mat( Data['X'] )
    lprPerItem = logsumexp( lpr, axis=1 )
    resp   = np.exp( lpr-lprPerItem[:,np.newaxis] ) 
    LP['resp'] = resp
    return LP
    
  def calc_evidence( self, Data, SS, LP ):
    return self.E_logpX( LP, SS) \
           + self.E_logpZ( LP ) - self.E_logqZ( LP ) \
           + self.E_logpW()   - self.E_logqW() \
           + self.E_logpPhi() - self.E_logqPhi()
           
  def get_global_suff_stats( self, Data, LP ):
    ''' 
    '''
    SS = dict()
    SS['N'] = np.sum( LP['resp'], axis=0 )
    SS = self.get_obs_suff_stats( SS, Data, LP )
    return SS
    
  def update_global_params( self, SS ):
    '''
    '''
    self.alpha   = self.alpha0 + SS['N']
    self.Elogw      = digamma( self.alpha ) - digamma( self.alpha.sum() )
    self.update_obs_params( SS )
    
    
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
