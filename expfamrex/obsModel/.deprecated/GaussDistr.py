'''
  Gaussian distribution in D-dimensions
    for generating real-valued vectors
    
  Following Bishop, we can parameterize the distribution
    with the precision matrix invSigma (the inverse of Sigma)
    instead of Sigma itself.

  This leads to cleaner updates, and the code its optimized so that
    we never explicitly perform matrix inversion, instead using
    more numerically stable routines to compute relevant quantities.

  Parameters
  -------
    Sigma  :  DxD matrix
    
    mu     :  Dx1 vector, mean of the distribution
  
'''
import numpy as np
import scipy.linalg
from scipy.special import digamma, gammaln

LOGPI = np.log(np.pi)
LOGTWO = np.log(2.00)
LOGTWOPI = np.log( 2.0*np.pi )
EPS = 10*np.finfo(float).eps

np.set_printoptions( linewidth=120, precision=4)

class GaussDistr(object):
 
  def disp(self):
    print self.mu
    if not self.doInv:
      print self.Sigma

  def __init__(self, mu=None, Sigma=None, invSigma=None):
    self.mu   = np.asarray(mu)
    self.D = self.mu.size

    if Sigma is not None:
      self.Sigma = Sigma
      self.doInv = False 
      self.cholSigma = scipy.linalg.cholesky( self.Sigma, lower=True )
      self.logdetSigma = 2*np.sum( np.log( np.diag( self.cholSigma )))
    else:
      self.invSigma = invSigma
      self.doInv = True
      self.cholinvSigma = scipy.linalg.cholesky( self.invSigma, lower=True )
      self.logdetSigma = -2*np.sum( np.log( np.diag( self.cholinvSigma )))
    
  def getMean( self ):
    return self.mu  
    
  def getMAP( self ):
    return self.mu
    
  def getPosteriorParams( self, SS, k ):
    return GaussWishDistr( D=self.D, mu=SS['mean'][k], Sigma=SS['covar'][k] )
    
  def logNormConst( self ):
    '''Calculate log normalization constant of self
    '''
    return -0.5*self.D*LOGTWOPI - 0.5*self.logdetSigma

  def log_pdf( self, X ):
    if type(X) is dict:
      X = X['X']
    return self.logNormConst() - 0.5*self.dist_mahalanobis( X )
   
  def dist_mahalanobis(self, X):
    '''Calculate Mahalanobis distance to x
             dist(x) = (x-m)'*W*(x-m)
       If X is a matrix, we compute a vector of distances to each row vector
             dist(X)[n] = (x_n-m)'*W*(x_n-m)
    '''
    if self.doInv:
      Q = np.dot( self.cholinvSigma, (X-self.mu).T )
    else:
      Q = scipy.linalg.solve_triangular( self.cholSigma, (X-self.mu).T,lower=True)
    return np.sum( Q**2, axis=0)


###############################################################################
def np2flatstr( X, fmt='% .6f' ):
  return ' '.join( [fmt % x for x in X.flatten() ] )  
  
     
