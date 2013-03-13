'''
  Represents classic Bayesian mixture model
    with a finite number of components K

 Author: Mike Hughes (mike@michaelchughes.com)

 Parameters
 -------
   alpha0   : scalar hyperparameter of symmetric Dirichlet prior on mix. weights
   obsPrior : Python object that represents prior on emission params
   
 Usage
 -------
   gmm = GMM( K=10, alpha0=0.1, obsPrior=gw )

 Inference
 -------
   See VBLearner.py  or EMLearner.py

 References
 -------
   Pattern Recognition and Machine Learning, by C. Bishop.
'''
import numpy as np
from ..util.MLUtil import logsumexp, np2flatstr, flatstr2np

class MixModel( object ):

  def __init__( self, K, alpha0, **kwargs ):
    self.qType = 'EM'
    self.K = K
    self.alpha0 = alpha0
    self.w = np.zeros( K )
    # obs distr specfics left up to sub-classes

  def get_info_string( self):
    return 'Finite mixture model with %d components' % (self.K)
    
  def to_string( self ):
    return np2flatstr( self.w )

  def calc_local_params( self, Data, LP ):
    ''' 
    '''
    lpr = np.log( self.w ) + LP['log_soft_ev']
    lprPerItem = logsumexp( lpr, axis=1 )
    resp   = np.exp( lpr-lprPerItem[:,np.newaxis] ) 
    LP['resp'] = resp
    LP['evidence'] = lprPerItem.sum()
    assert np.allclose( np.sum(resp,axis=1), 1.0 )
    return LP

  def calc_evidence( self, Data, SS, LP ):
    return LP['evidence']

  def get_global_suff_stats( self, Data, SS, LP ):
    ''' 
    '''
    SS['N'] = np.sum( LP['resp'], axis=0 )
    return SS

  def update_global_params( self, SS, rho=None, **kwargs ):
    '''
    '''
    w = self.alpha0 + SS['N']
    w /= w.sum()
    if rho is None:
    	self.w = w
    else:
    	self.w = rho*w + (1-rho)*self.w
