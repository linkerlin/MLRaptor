'''
  GaussWishDistrIndep.py
  Independently factorized Joint Distribution over 
     mean vector mu, precision matrix Lambda

  Lambda ~ Wish( dF, invW )
  mu ~ Normal( m,  L )
  
  Parameters
  -------
    m,L    define mu's prior. See GaussianDistr.py.
    v,invW define Lambdas's prior. See WishartDistr.py.
'''
import numpy as np

from .GaussianDistr import GaussianDistr
from .WishartDistr import WishartDistr

LOGTWOPI = np.log( 2.0*np.pi )

class GaussWishDistrIndep( object ):

  def __init__( self, muD=None, LamD=None ):
    self.muD  = muD
    self.LamD = LamD

  def set_dims( self, D ):
    if self.muD is None:
      self.muD = GaussianDistr()
      self.LamD = WishartDistr()
    self.muD.set_dims( D)
    self.LamD.set_dims( D)

  def to_dict( self ):
    return dict( v=self.LamD.v, invW=self.LamD.invW, m=self.muD.m, L=self.muD.L )

  def E_log_pdf( self, X):
    '''
      E[ log p(X | this distribution )], up to prop. constant
      Returns N-dim vector, one entry per row of X
    '''
    logp = 0.5*self.LamD.E_logdetLam() 
    logp -= 0.5*self.E_dist_mahalanobis( X )
    return logp

  def E_dist_mahalanobis( self, X ):
    ''' Given NxD matrix X,
          compute Nx1 vector of distances from this distribution
        Dist[n] = E[ (X[n]-mu)^T *Lam* (X[n]-mu)  ]
    '''
    Xdiff = X - self.muD.m
    Dist  = self.LamD.E_dist_mahalanobis( Xdiff.T )
    return Dist + self.LamD.E_traceLambda( self.muD.get_invL() )

  def getPosteriorDistr( self, N, x, xxT, ELam ):
    '''
    '''
    for rep in xrange( 5 ):
      muD  = self.muD.getPosteriorDistr( N, x, ELam )
      LamD = self.LamD.getPosteriorDistr( N, x, xxT, muD.m, muD.get_covar() )
      if rep > 0 and np.allclose( ELam, prev ):
        break
      prev = ELam
      ELam = LamD.E_Lam()
    return GaussWishDistrIndep( muD, LamD )
