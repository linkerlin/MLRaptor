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
    if phi is None:
      self.phi = None
      return
    self.phi   = np.asarray(phi)
    self.D = self.phi.size
    self.set_helpers()

  def set_dims(self, D ):
    self.phi = np.ones( D )
    self.D = D
    self.set_helpers()
    
  def set_helpers( self):
    self.logphi = np.log( self.phi)
    self.log1mphi = np.log(1-self.phi)
    
  def get_log_norm_const( self ):
    '''Calculate log normalization constant
    '''
    return 0.0

  def log_pdf( self, X ):
    if type(X) is dict:
      X = X['X']
    return np.sum( X*self.logphi + (1-X)*self.log1mphi, axis=1)

  ###########################################  Useful for inspecting
  def getMean( self ):
    return self.phi  
    
  def getMAP( self ):
    return np.asarray( self.phi > 0.5, dtype=self.phi.dtype )
    
###############################################################################
def np2flatstr( X, fmt='% .6f' ):
  return ' '.join( [fmt % x for x in X.flatten() ] )  
  
     
