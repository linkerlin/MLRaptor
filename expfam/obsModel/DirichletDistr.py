'''
  Dirichlet distribution in D-dimensions
    
  Parameters
  -------
    lamvec  :  Dx1 vector, 0 < lamvec[d]
  
'''
import numpy as np
import scipy.linalg
from scipy.special import digamma, gammaln


class DirichletDistr(object):

  def __init__(self, lamvec=None):
    self.lamvec = lamvec
    if lamvec is not None:
      self.D = lamvec.size
      self.set_helpers()

  def set_dims( self, D ):
    self.D = D
    self.lamvec = 1.0 * np.ones( D )
    self.set_helpers()
    
  def set_helpers(self):
    self.lamsum = self.lamvec.sum()
    self.digammalamvec = digamma(self.lamvec)
    self.digammalamsum = digamma(self.lamsum)
    self.Elogphi   = self.digammalamvec - self.digammalamsum

  def getPosteriorDistr( self, TermCountVec ):
    return DirichletDistr( self.lamvec + TermCountVec )

  def get_log_norm_const( self ):
    ''' Returns log( 1/Z ) = -log(Z), 
         where p( phi | a,b) = 1/Z(a,b) f(phi)
    '''
    return gammaln( self.lamsum ) - np.sum(gammaln(self.lamvec ))

  def get_entropy(self):
    H = -1*self.get_log_norm_const()
    H -= np.sum( (self.lamvec-1)*self.Elogphi )
    return H
    
  def E_log_pdf( self, Data ):
    '''
    '''
    try:
      return self.log_pdf_from_dict( Data )
    except KeyError:
      return self.log_pdf_from_mat( Data['X'] )

  def log_pdf_from_dict( self, Data ):
    lpr = np.zeros( Data['nObsEntry'] )
    tokenID=0
    for docCDict in Data['BoW']:
      for (wID,count) in docCDict.items():
        lpr[tokenID] = count*self.Elogphi[wID]
        assert count == Data['countvec'][tokenID]
        tokenID += 1
    return lpr
