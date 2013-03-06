'''
  Wishart distribution in D-dimensions
    for generating positive definite prec. matrices

    \Lambda ~ Wish( deg free=v, scale mat=W^{-1} )

    E[ \Lambda ] = v W

  Parameters
  -------    
    v     :  positive real scalar
    invW  :  DxD positive definite matrix
'''
import numpy as np

import scipy.linalg
from scipy.special import gammaln, digamma

LOGPI = np.log(np.pi)
LOGTWO = np.log(2.00)
LOGTWOPI = np.log( 2.0*np.pi )

EPS = 10*np.finfo(float).eps

def MVgammaln( x, D ):
  return 0.25*D*(D-1)*LOGPI \
          + gammaln( 0.5*(x+1 - np.arange(1,D+1)) ).sum()

def MVdigamma( x, D ):
  return digamma( 0.5*( x+1 - np.arange(1,D+1)) ).sum()

class WishartDistr( object ):

  def __init__(self, v=None, invW=None):
    self.v = v
    self.invW = invW
    if self.v is not None:
      self.D = invW.shape[0]
      self.set_helper_params()
 
  def set_dims( self, D):
    self.D = D
    if self.D is not None:
      self.init_params()
   
  def init_params( self ):
    self.v    = self.D+2
    self.invW = np.eye( self.D )
    self.set_helper_params()
    
  def set_helper_params(self):
    self.cholinvW = scipy.linalg.cholesky( self.invW , lower=True )
    self.logdetW  = -2.0*np.sum( np.log( np.diag( self.cholinvW ) )  )

  ########################################## Accessors
  def E_Lam(self):
    return self.v * np.linalg.solve( self.invW, np.eye(self.D) )

  def E_logdetLam( self ):
    return self.logdetW + self.D*LOGTWO + MVdigamma(self.v, self.D)

  ########################################## Posterior calc
  def getPosteriorDistr( self, EN, Ex, ExxT, Emu, Ecov ):
    v    = self.v + EN
    invW = self.invW + ExxT - 2*np.outer(Ex, Emu) + EN*np.outer(Emu,Emu) + EN*Ecov
    return WishartDistr( v, invW )

  def get_log_norm_const( self ):
    ''' p( Lambda ) = 1/Z * f(mu)
        this returns log( 1/Z) = -1*log( Z )
    '''
    return -0.5*self.v*self.logdetW -0.5*self.v*self.D*LOGTWO  - MVgammaln( self.v, self.D )

  def get_entropy( self ):
    '''
      Returns H[ p(Lambda) ] = -1*\int p(Lam) log(p(Lam)) dLam
    '''
    return -1*self.get_log_norm_const() - 0.5*(self.v -self.D-1)*self.E_logdetLam() + 0.5*self.v*self.D

  ########################################## Useful expectations
  def E_dist_mahalanobis(self, dX ):
    '''Calculate Mahalanobis distance to x
             dist(x) = dX'*E[Lambda]*dX
       If X is a matrix, we compute a vector of distances to each row vector
             dist(X)[n] = dX[n]'*E[Lambda]*dX[n]
    '''
    Q = scipy.linalg.solve_triangular( self.cholinvW, dX,lower=True)
    return self.v * np.sum( Q**2, axis=0)

  def E_traceLambda( self, S):
    '''Calculate trace( S* E[Lambda] ) in numerically stable way
          = v * trace( S* W ), without explicitly inverting W
    '''
    try:
      U = scipy.linalg.cholesky( S , lower=True )
    except scipy.linalg.LinAlgError:
      U = scipy.linalg.cholesky( S + EPS*np.eye( self.D), lower=True ) 
    Q = scipy.linalg.solve_triangular( self.cholinvW, U, lower=True)
    return self.v * np.sum(Q**2)
