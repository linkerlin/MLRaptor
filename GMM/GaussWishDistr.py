'''
  Gaussian-Wishart distribution in D-dimensions
    for generating mean vector/precision matrix pairs
    
    prec matrix Lam ~ Wish( dF, W )
    mean vector  mu ~ Normal( m, inv(kappa*Lam)  )
  
  which means mu has covariance
    Sigma = (kappa*Lam)^-1

  or equivalently,
    Sigma/kappa ~ InverseWishart( dF, inv(W)  )

  following Bishop, we parameterize the distribution with 
    the scale matrix invW (the inverse of W) instead of W itself.

  This leads to cleaner updates, and the code its optimized so that
    we never explicitly perform matrix inversion, instead using
    more numerically stable routines to compute relevant quantities.

  Parameters
  -------
    dF    : scalar , degrees of freedom of Wishart distr.
    invW  : DxD positive definite matrix for Wishart distr.
              mean(Lam) = dF*W
    
    m     :  Dx1 vector, mean of the Gaussian on mu
    kappa :  scalar, additional precision parameter of Gaussian
              covar(m) = inv(Lam)/kappa             
  
'''
import numpy as np
import scipy.linalg
from scipy.special import digamma, gammaln
LOGPI = np.log(np.pi)
LOGTWO = np.log(2.00)
LOGTWOPI = np.log( 2.0*np.pi )
EPS = np.finfo(float).eps

def np2flatstr( X, fmt='% .6f' ):
  return ' '.join( [fmt % x for x in X.flatten() ] )  
  
class GaussWishDistr(object):
 
  def __init__(self, dF=None, invW=None, kappa=None, m=0.0, D=None):
   if kappa is None and dF is None and invW is None:
     if D is None: raise ArgumentError()
     kappa = 1.0
     dF    = D + 2
     invW  = np.eye( D )
   self.kappa = kappa
   self.invW = invW
   self.dF   = dF
   self.D    = D
   if self.D is None:
     self.D = invW.shape[0]
   assert self.D == invW.shape[0]
   muMean =np.asarray( m )
   if muMean.size == self.D:
     self.m    = muMean
   elif muMean.size ==1:
     self.m    = np.tile( muMean, (self.D) )
   self.cholinvW = scipy.linalg.cholesky( invW, lower=True ) 
   self.logdetW     = self.log_W_det()
   
  def __str__(self): 
    return '%f %s %f %s' % (self.kappa,  np2flatstr(self.m), self.dF, np2flatstr(self.invW)  )
    
  def getMean( self ):
    mu = self.m
    Sigma = self.invW / ( self.dF - self.D - 1 )
    return mu,Sigma  
    
  def getMAP( self ):
    assert self.dF > self.D+1
    muMAP = self.m
    SigmaMAP = self.invW / (self.dF + self.D + 1 )
    return muMAP, SigmaMAP
    
  def getPosteriorParams( self, N, mean, covar ):
    kappa = self.kappa+N
    m = ( self.kappa*self.m + N*mean ) / kappa
    invW  = self.invW + N*covar  \
            + (self.kappa*N)/(self.kappa+N)*np.outer(mean - self.m,mean - self.m)
    return GaussWishDistr( self.dF+N, invW,  kappa, m )
    
  def log_W_det( self ):
    return -2.0*np.sum( np.log( np.diag( self.cholinvW ) )  )  
  
  def entropyWish(self):
    '''Calculate entropy of this Wishart distribution,
         as defined in Bishop PRML B.82
    '''
    return -self.logWishNormConst() \
           - 0.5*(self.dF-self.D-1)*self.ElogdetLam() \
           + 0.5*self.D*self.dF
  
  def ElogdetLam( self ):
    '''Calculate expected determinant of the precision matrix drawn from
        this Wishart distribution, as defined in Bishop PRML B.81
       E[ log |Lam| ] = D*log(2) + log|W| + \sum_{d=1}^D digamma( (dF+1-d)/2 )
    '''
    return digamma( 0.5*(self.dF+1 - np.arange(1,self.D+1)) ).sum() \
           + self.D*LOGTWO  + self.logdetW
    
  def logWishNormConst( self ):
    '''Calculate normalization constant of self's Wishart distribution
       Wish( Lam | dF, W ) = B(dF,W)*|Lam|^(dF-D-1)/2 exp( -.5 tr( invW*Lam )
         where the norm constant is B() is defined in Bishop PRML B.79
            B(dF,W) = |W|^{-dF/2} 2^{-dF*D/2} pi^{-D(D-1)/4} 
                        / prod_{d=1}^D  Gam( (dF+1 - d) / 2 )
    '''
    logB = -0.5 *self.dF * self.logdetW \
          -0.5 *self.dF*self.D* LOGTWO  \
          -0.25*self.D*(self.D-1)*LOGPI \
          - gammaln( 0.5*(self.dF+1 - np.arange(1,self.D+1)) ).sum()
    return logB
   
  def dist_mahalanobis(self, X):
    '''Calculate Mahalanobis distance to x
             dist(x) = (x-m)'*W*(x-m)
       If X is a matrix, we compute a vector of distances to each row vector
             dist(X)[n] = (x_n-m)'*W*(x_n-m)
    '''
    # Method: Efficient solve via cholesky
    #   let dx  = x-m, a Dx1 vector
    #   want to solve  dist(x) = dx'*W*dx
    #   let L'L = W,   dist(x) = dx'L' L dx = q'*q = \sum_d q[d]^2
    #     where q is a Dx1 vector.            
    dX = (X-self.m[np.newaxis,:]).T  #  D x N
    Q = scipy.linalg.solve_triangular( self.cholinvW, dX,lower=True)
    return np.sum( Q**2, axis=0)
  
  def traceW( self, S):
    '''Calculate trace( S* self.W ) in numerically stable way
    '''
    U = scipy.linalg.cholesky( S + EPS*np.eye(self.D) , lower=True )
    Q = scipy.linalg.solve_triangular( self.cholinvW, U, lower=True)
    return np.sum(Q**2)
     
