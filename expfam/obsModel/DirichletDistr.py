'''
  Dirichlet distribution in D-dimensions
    
  Parameters
  -------
    lamvec  :  Dx1 vector, 0 < lamvec[d]
  
'''
import numpy as np
import scipy.linalg
import itertools
from scipy.special import digamma, gammaln

class DirichletDistr(object):

  def __init__(self, lamvec=None):
    self.lamvec = lamvec
    if lamvec is not None:
      self.D = lamvec.size
      self.set_helpers()

  def set_dims( self, D ):
    self.D = D
    self.lamvec = 1.0/D * np.ones( D )
    self.set_helpers()
    
  def set_helpers(self):
    self.lamsum = self.lamvec.sum()
    self.digammalamvec = digamma(self.lamvec)
    self.digammalamsum = digamma(self.lamsum)
    self.Elogphi   = self.digammalamvec - self.digammalamsum

  def rho_update( self, rho, starDistr ):
    self.lamvec = rho*starDistr.lamvec + (1.0-rho)*self.lamvec
    self.set_helpers()

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
      Data['wordIDs_perGroup'][0]
      return self.log_pdf_from_list( Data )
    except KeyError:
      return np.dot( Data['X'], self.Elogphi )

  def log_pdf_from_list( self, Data ):
    lpr = np.zeros( Data['nObs'] )
    for docID in xrange( Data['nGroup'] ):
      lpr[  Data['GroupIDs'][docID][0]:Data['GroupIDs'][docID][1] ] = self.Elogphi[:, Data['wordIDs_perGroup'][docID] ].T
    return lpr
    '''
    tokenID=0
    for docID in xrange( Data['nGroup'] ):
      for (wID,count) in itertools.izip( Data['wordIDs_perGroup'][docID], Data['wordCounts_perGroup'][docID] ):
        lpr[tokenID] = count*self.Elogphi[wID]
        tokenID += 1
    return lpr
    '''

  def log_pdf_from_dict( self, Data ):
    lpr = np.zeros( Data['nObsEntry'] )
    tokenID=0
    for docCDict in Data['BoW']:
      for (wID,count) in docCDict.items():
        lpr[tokenID] = count*self.Elogphi[wID]
        assert count == Data['countvec'][tokenID]
        tokenID += 1
    return lpr
