'''
  Wishart distribution in D-dimensions
    for generating positive definite prec. matrices

    \Lambda ~ Wish( deg free=v, scale mat=W^{-1} )

    E[ \Lambda ] = v W

  Parameters
  -------    
    v     :  positive real scalar
    invW  :  DxD positive definite matrix
'''

class GaussianDistr( object ):

  def __init__(self, m, L):
    self.m = m
    self.L = L
    self.D = m.size
    assert np.all(L.shape == self.D)
    
  def set_dims( self, D):
    self.D = D
    if self.D is not None:
      self.init_params()
   
  def init_params( self ):
    self.L = np.eye( self.D )
    self.m = np.zeros( (self.D) )
    self.set_helper_params()
    
  def set_helper_params(self):
    self.cholL   = scipy.linalg.cholesky( self.cholL , lower=True )
    self.logdetL = 2.0*np.sum( np.log( np.diag( self.cholL ) )  )

  ########################################## Accessors
  def get_mean(self):
    return self.m

  def get_covar(self):
    try:
      return self.invL
    except Exception:
      self.invL = np.inv( self.L )
      return self.invL

  ########################################## Posterior calc
  def getPosteriorDistr( self, EN, Esum, ELam ):
    L = self.L + EN*ELam
    m = np.dot(self.L,self.m) + np.dot( ELam, Esum )
    m = np.solve( L, m )
    return GaussianDistr( m, L )

  def get_log_norm_const( self ):
    ''' p( mu ) = 1/Z * f(mu)
        this returns -1*log( Z )
    '''
    return -0.5*self.D*LOGTWOPI + 0.5*self.logdetL

  def get_entropy( self ):
    return log_norm_const() + -0.5*self.D
