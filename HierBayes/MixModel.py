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
from MLUtil import logsumexp

class MixModel( object ):

  def __init__( self, K, alpha0 ):
    self.K = K
    self.alpha0 = 0.0
    self.w = np.zeros( K )
    # obs distr specfics left up to sub-classes

  def calc_local_params( self, Data ):
    ''' 
    '''
    LP = dict()
    lpr = np.log( self.w ) + self.calc_soft_evidence_mat( Data['X'] )
    lprPerItem = logsumexp( lpr, axis=1 )
    resp   = np.exp( lpr-lprPerItem[:,np.newaxis] ) 
    LP['resp'] = resp
    LP['evidence'] = lprPerItem.sum()
    return LP

  def calc_evidence( self, Data, SS, LP ):
    return LP['evidence']

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
    self.w = self.alpha0 + SS['N']
    self.w /= self.w.sum()
    self.update_obs_params( SS )
