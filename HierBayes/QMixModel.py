'''
  Represents mean-field factorization of a 
    Bayesian mixture model with a finite number of components K

 Author: Mike Hughes (mike@michaelchughes.com)

 Parameters
 -------
   alpha0   : scalar hyperparameter of symmetric Dirichlet prior on mix. weights
   obsPrior : Python object that represents prior on emission params
                  conventionally, obsPrior = Gaussian-Wishart distribution 
                    see GaussWishDistr.py
   
 Usage
 -------
   gw   = GaussWishDistr( dF=3, invW=np.eye(2)  )
   qgmm = MixModel( K=10, alpha0=0.1, obsPrior=gw )

 Inference
 -------
   See VBLearner.py  or EMLearner.py

 References
 -------
   Pattern Recognition and Machine Learning, by C. Bishop.
'''

class QMixModel( object ):

  def __init__( K, alpha0, obsPrior=None ):
    self.K = K
    self.alpha0 = alpha0

    self.obsPrior = obsPrior
    self.obsDistrList = [obsPrior for k in xrange(K)]


  def get_global_suff_stats( self, Data, LP ):
    ''' 
    '''
    SS = dict()
    SS['N'] = np.sum( LP['resp'], axis=0 )
    
  def update_global_params( self, SS ):
    '''
    '''
    self.w = self.alpha0 + SS['N']

    for k in xrange( self.K ):
      self.obsDistrList[k] = self.obsPrior.getPosteriorDistr( SS, k )    

