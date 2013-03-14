'''
  Multinomial distribution over symbol set of size V
    for generating count data
    
  Parameters
  -------
    phi  :  Vx1 vector, 0 <= phi[v] <= 1
                        \sum_{v=1}^V \phi[v] = 1 (sums to one)
  
'''
import numpy as np
import scipy.linalg
from scipy.special import digamma, gammaln

class MultinomialDistr(object):
 
  def disp(self):
    print self.phi

  def __init__(self, phi=None):
    if phi is None:
      self.phi = None
      return
    self.phi   = np.asarray(phi)
    self.V = self.phi.size
    self.set_helpers()

  def set_dims(self, V ):
    self.phi = np.ones( V )
    self.V = V
    self.set_helpers()
    
  def set_helpers( self):
    self.logphi = np.log( self.phi)
    
  def get_log_norm_const( self ):
    '''Calculate log normalization constant
    '''
    return 0.0

  def log_pdf( self, Data ):
    '''
    '''
    try:
      return log_pdf_from_dict( Data['BoW'] )
    except KeyError:
      return log_pdf_from_mat( Data['X'] )

  def log_pdf_from_dict( self, BoW ):
    lpr = np.zeros( Data['nObs'] )
    tokenID=0
    for docCDict in BoW:
      for (wID,count) in docCDict:
        lpr[tokenID] = count*self.logphi[wID]
        tokenID += 1
    return lpr
 
  def log_pdf_from_mat( self, X ):
    return np.sum( X*self.logphi, axis=1)

    
###############################################################################
def np2flatstr( X, fmt='% .6f' ):
  return ' '.join( [fmt % x for x in X.flatten() ] )  
  
     
