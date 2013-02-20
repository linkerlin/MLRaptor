'''
  Bernoulli distribution in D-dimensions
    for generating binary vectors.
    
  Parameters
  -------
    phi  :  Dx1 vector, 0 <= phi[d] <= 1
  
'''
import numpy as np
import scipy.linalg
from scipy.special import digamma, gammaln

np.set_printoptions( linewidth=120, precision=4)

class BernoulliDistr(object):
 
  def disp(self):
    print self.phi

  def __init__(self, phi=None):
    self.phi   = np.asarray(phi)
    self.D = self.phi.size

    self.logphi = np.log(phi)
    self.log1mphi = np.log(1-phi)
    
  def getMean( self ):
    return self.phi  
    
  def getMAP( self ):
    return np.asarray( self.phi > 0.5, dtype=self.phi.dtype )
    
  def getPosteriorParams( self, SS, k ):
    return BernoulliDistr( phi=SS['countvec'][k] )
    
  def logNormConst( self ):
    '''Calculate log normalization constant of self
    '''
    return 0.0

  def log_pdf( self, X ):
    if type(X) is dict:
      X = X['X']
    return X*self.logphi + (1-X)*self.log1mphi 

###############################################################################
def np2flatstr( X, fmt='% .6f' ):
  return ' '.join( [fmt % x for x in X.flatten() ] )  
  
     
