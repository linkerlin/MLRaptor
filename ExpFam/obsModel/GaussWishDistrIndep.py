'''
  Prior parameterization for mean mu, precision Lambda

    prec matrix Lam ~ Wish( dF, invW )
    mean vector  mu ~ Normal( m,  L )
  
  Parameters
  -------
    m,L    define mu's prior (see GaussianDistr2.py)
    v,invW define Lam's prior (see WishartDistr.py)
'''
import numpy as np

from .GaussianDistr2 import GaussianDistr2
from .WishartDistr import WishartDistr

LOGTWOPI = np.log( 2.0*np.pi )

class GaussWishDistrIndep( object ):

  def __init__( self, muD=None, LamD=None ):
    self.muD  = muD
    self.LamD = LamD

  def set_dims( self, D ):
    if self.muD is None:
      self.muD = GaussianDistr2()
      self.LamD = WishartDistr()
    self.muD.set_dims( D)
    self.LamD.set_dims( D)

  def E_log_pdf( self, X):
    '''
      E[ log p(X | this distribution )], up to prop. constant
    '''
    logp = 0.5*self.LamD.E_logdetLam() \
          -0.5*self.E_dist_mahalanobis( X )
    return logp

  def E_dist_mahalanobis( self, X ):
    ''' Given NxD matrix X,
          compute Nx1 vector of distances from this distribution
        Dist[n] = E[ (X[n]-mu)^T *Lam* (X[n]-mu)  ]
    '''
    Xdiff = X - self.muD.m
    Dist  = self.LamD.E_dist_mahalanobis( Xdiff.T )
    return Dist + self.LamD.E_traceLambda(self.muD.invL)

  def getPosteriorDistr( self, N, x, xxT):
    '''
    '''

    ELam = self.LamD.E_Lam()
    for rep in xrange( 5 ):
      muD  = self.muD.getPosteriorDistr( N, x, ELam )
      LamD = self.LamD.getPosteriorDistr( N, x, xxT, muD.m, muD.get_covar() )
      if rep > 0 and np.allclose( ELam, prev ):
        break
      prev = ELam
      ELam = LamD.E_Lam()
    return GaussWishDistrIndep( muD, LamD )
