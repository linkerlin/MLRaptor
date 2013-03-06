'''
  Beta distribution in D-dimensions
    
  Parameters
  -------
    a  :  Dx1 vector, 0 < a[d]
    b  :  Dx1 vector, 0 < b[d]
  
'''
import numpy as np
import scipy.linalg
from scipy.special import digamma, gammaln

np.set_printoptions( linewidth=120, precision=4)

nON_DEF  = 0.25
nOFF_DEF = 1

class BetaDistr(object):
 
  def disp(self):
    print self.a/(self.a+self.b)

  def __init__(self, a=None, b=None):
    if a is None:
      self.a = None
      return
    self.a   = np.asarray(a)
    self.b   = np.asarray(b)
    self.D = self.b.size
    self.set_helpers()
    
  def set_dims( self, D ):
    self.D = D
    self.a = nON_DEF* np.ones( D )
    self.b = nOFF_DEF * np.ones( D )
    self.set_helpers()
    
  def set_helpers(self):
    DENOM = digamma( self.a+self.b )
    self.Elogphi   = digamma(self.a) - DENOM
    self.Elog1mphi = digamma(self.b) - DENOM
    self.Ephi = self.a/(self.a+self.b)
   
  def getPosteriorDistr( self, N, CountON ):
    aPost = self.a +  CountON
    bPost = self.b +  N-CountON
    return BetaDistr( aPost, bPost)
    
  def get_log_norm_const( self ):
    ''' Returns log( 1/Z ) = -log(Z), 
         where p( phi | a,b) = 1/Z(a,b) f(phi)
    '''
    return np.sum( gammaln( self.a+self.b ) - gammaln(self.a) - gammaln(self.b) )

  def get_entropy(self):
    H = -1*self.get_log_norm_const()
    H -= np.sum( (self.a-1)*digamma(self.a) )
    H -= np.sum( (self.b-1)*digamma(self.b) )
    H += np.sum( (self.a+self.b-2)*digamma(self.a+self.b) )
    return H
    
  def E_log_pdf( self, X ):
    if type(X) is dict:
      X = X['X']
    lpr = np.sum( X*self.Elogphi + (1-X)*self.Elog1mphi, axis=1)
    return lpr
    
###############################################################################
def np2flatstr( X, fmt='% .6f' ):
  return ' '.join( [fmt % x for x in X.flatten() ] )  
  
     
