'''
  Gaussian distribution in D-dimensions
    for generating mean vectors

    \mu ~ Normal( mean=m, covar=L^{-1} )

  Parameters
  -------    
    m     :  Dx1 vector
    L     :  DxD inverse covariance matrix
'''
import numpy as np
import scipy.linalg

LOGPI = np.log(np.pi)
LOGTWO = np.log(2.00)
LOGTWOPI = np.log( 2.0*np.pi )

EPS = 10*np.finfo(float).eps

class GaussianDistr2( object ):

  def __init__(self, m=None, L=None):
    if L is not None:
      self.L = np.asarray( L )
      self.m = np.asarray( m )  
      self.D = self.m.size
      assert self.L.size == self.D**2
      self.set_helper_params()
    else:
      self.m = m
      self.L = L

  def set_dims( self, D):
    self.D = D
    if self.D is not None:
      self.init_params()

  def init_params( self ):
    self.L = 0.01 * np.eye( self.D ) # small prec = large variance
    self.m = np.zeros( (self.D) )
    self.set_helper_params()
    
  def set_helper_params(self):
    self.cholL   = scipy.linalg.cholesky( self.L )
    self.logdetL = 2.0*np.sum( np.log( np.diag( self.cholL ) )  )
    self.invL = np.linalg.solve( self.L, np.eye(self.D) )

  ########################################## Accessors
  def get_mean(self):
    return self.m

  def get_covar(self):
    try:
      return self.invL
    except Exception:
      self.invL = np.linalg.pinv( self.L )
      return self.invL

  ########################################## Posterior calc
  def getPosteriorDistr( self, EN, Esum, ELam ):
    L = self.L + EN*ELam
    m = np.dot(self.L,self.m) + np.dot( ELam, Esum )
    m = np.linalg.solve( L, m )
    return GaussianDistr2( m, L )

  def get_log_norm_const( self ):
    ''' p( mu ) = 1/Z * f(mu)
        this returns -1*log( Z )
    '''
    return -0.5*self.D*LOGTWOPI + 0.5*self.logdetL

  def get_entropy( self ):
    '''   Returns H[ p(mu) ] = -1*\int p(mu) log(p(mu)) dLam
    '''
    return -1*self.get_log_norm_const() + 0.5*self.D

  ########################################## Soft evidence computation
  def log_pdf( self, X ):
    if type(X) is dict:
      X = X['X']
    return self.get_log_norm_const() - 0.5*self.dist_mahalanobis( X )
   
  def dist_mahalanobis(self, X):
    '''  Given NxD matrix X, compute  Nx1 vector Dist
            Dist[n] = ( X[n]-m )' L (X[n]-m)
    '''
    Q = np.dot( self.cholL, (X-self.m).T )
    return np.sum( Q**2, axis=0)
