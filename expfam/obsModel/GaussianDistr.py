'''
  Gaussian distribution in D-dimensions

  x ~ Normal( mean=m, covar=L^{-1} )

  Parameters
  -------    
    m     :  Dx1 mean vector 
    L     :  DxD inverse covariance matrix
'''
import numpy as np
import scipy.linalg

from ..util.MLUtil import np2flatstr, flatstr2np, dotATA, dotABT, dotATA

LOGPI = np.log(np.pi)
LOGTWO = np.log(2.00)
LOGTWOPI = np.log( 2.0*np.pi )

EPS = 10*np.finfo(float).eps

class GaussianDistr( object ):

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
    # Need lower chol if working with covar mat,
    #  but here can use upper since working with prec mat
    self.cholL   = scipy.linalg.cholesky( self.L ) #UPPER by default
    self.logdetL = 2.0*np.sum( np.log( np.diag( self.cholL ) )  )

  def get_invL( self ):
    try:
      return self.invL
    except Exception:
      self.invL = np.linalg.inv( self.L )
      return self.invL

  #######################################################  To/From String
  def from_string( self, mystr ):
    myvec = flatstr2np(  mystr )
    self.D = myvec[0]
    self.m = myvec[1:self.D+1]
    self.L = myvec[self.D+1:]
    self.L = np.reshape( self.L, (self.D, self.D) )

  def to_string( self ):
    return np2flatstr( np.hstack([self.D,self.m]) ) + np2flatstr( self.L )

  def to_dict( self ):
    return dict( m=self.m, L=self.L )

  #######################################################   ExpFam Natural Param Convert
  def get_natural_params( self ):
    eta = self.L, np.dot(self.L,self.m)
    return eta 

  def set_natural_params( self, eta ):
    L, Lm = eta
    self.L = L
    self.m = np.linalg.solve( L, Lm) # invL*L*m = m

  ##################################################### Accessors
  def get_mean(self):
    return self.m

  def get_covar(self):
    try:
      return self.invL
    except Exception:
      self.invL = np.linalg.inv( self.L )
      return self.invL

  #################################################### Posterior calc
  def rho_update( self, rho, newGaussDistr ):
    etaCUR = self.get_natural_params()
    etaSTAR = newGaussDistr.get_natural_params()
    etaNEW = list(etaCUR)
    for i in xrange(len(etaCUR)):
      etaNEW[i] = rho*etaSTAR[i] + (1-rho)*etaCUR[i]
    self.set_natural_params( etaNEW )
    self.set_helper_params()

  def getPosteriorDistr( self, EN, Esum, ELam ):
    L = self.L + EN*ELam
    Lm = np.dot(self.L,self.m) + np.dot( ELam, Esum )
    m = np.linalg.solve( L, Lm )
    return GaussianDistr( m, L )

  #################################################### Norm Constants
  def get_log_norm_const( self ):
    ''' p( mu ) = 1/Z * f(mu)
        this returns -1*log( Z )
    '''
    return -0.5*self.D*LOGTWOPI + 0.5*self.logdetL

  def get_entropy( self ):
    '''   Returns H[ p(mu) ] = -1*\int p(mu) log(p(mu)) dLam
    '''
    return -1*self.get_log_norm_const() + 0.5*self.D

  #################################################### Soft evidence computation
  def log_pdf( self, X ):
    if type(X) is dict:
      X = X['X']
    return self.get_log_norm_const() - 0.5*self.dist_mahalanobis( X )
  
  def dist_mahalanobis(self, X):
    '''  Given NxD matrix X, compute  Nx1 vector Dist
            Dist[n] = ( X[n]-m )' L (X[n]-m)
    '''
    Q = dotABT( self.cholL, X-self.m )
    #Q = np.dot( self.cholL, (X-self.m).T )
    return np.sum( Q**2, axis=0)
