lambdaDEF = 0.5

class DirichletDistr(object):

  def __init__(self, lamvec):
    self.lamvec = lamvec
    self.D = alphvec.size
    self.set_helpers()

  def set_dims( self, D ):
    self.D = D
    self.lamvec = lambdaDEF* np.ones( D )
    self.set_helpers()
    
  def set_helpers(self):
    self.lamsum = self.lamvec.sum()
    self.Elogphi   = digamma(self.lamvec) - digamma( self.lamsum )
    self.Ephi = self.lamvec/self.lamsum

  def getPosteriorDistr( self, TermCountVec ):
    return DirichletDistr( self.lamvec + TermCountVec )


  def get_log_norm_const( self ):
    ''' Returns log( 1/Z ) = -log(Z), 
         where p( phi | a,b) = 1/Z(a,b) f(phi)
    '''
    return np.sum( gammaln( self.lamvec ) - gammaln( self.lamsum )

  def get_entropy(self):
    H = -1*self.get_log_norm_const()
    H -= np.sum( (self.a-1)*digamma(self.a) )
    H -= np.sum( (self.b-1)*digamma(self.b) )
    H += np.sum( (self.a+self.b-2)*digamma(self.a+self.b) )
    return H
    
  def E_log_pdf( self, Data ):
    '''
    '''
    try:
      return log_pdf_from_dict( Data )
    except KeyError:
      return log_pdf_from_mat( Data['X'] )

  def log_pdf_from_dict( self, Data ):
    lpr = np.zeros( Data['nObs'] )
    tokenID=0
    for docCDict in Data['BoW']:
      for (wID,count) in docCDict:
        lpr[tokenID] = count*self.logphi[wID]
        tokenID += 1
    return lpr
