'''
 Mean-Field Variational Approximation
   to a Gaussian Mixture Model
    with a Dirichlet Process allocation
    
 Author: Mike Hughes (mike@michaelchughes.com)

 Parameters
 -------
   alpha0   : scalar hyperparameter of symmetric Dirichlet prior on mix. weights
   obsPrior : Python object that represents prior on emission params
                  conventionally, obsPrior = Gaussian-Wishart distribution 
                    see GaussWishDistr.py
   
 Usage
 -------
   gw   = GaussWishDistr( dF=3, invW=np.eye(2)  )
   qgmm = MixModel( K=10, alpha0=0.1, obsPrior=gw )

 Inference
 -------
   See VBLearnAlg.py

 References
 -------
   Pattern Recognition and Machine Learning, by C. Bishop.
'''
import QDPMixModel
import QDPGMM

import numpy as np

LOGPI = np.log(np.pi)
LOGTWO = np.log(2.00)
LOGTWOPI = np.log( 2.0*np.pi )

class QDPGMM( QDPMixModel.QDPMixModel ):

  def __init__( self, K=2, alpha0=None, obsPrior=None, **kwargs ):
    super(type(self),self).__init__( K, alpha0, **kwargs )
    self.obsPrior = obsPrior
    self.qobsDistr = [ None for k in xrange(self.K)]

  def set_dims( self, Data ):
    self.D = Data['X'].shape[1]
    
  def update_obs_params( self, SS, rho=None):
    ''' M-step update
    '''
    if rho is None:
      for k in xrange( self.K ):
        self.qobsDistr[k] = self.obsPrior.getPosteriorDistr( SS['N'][k], SS['mean'][k], SS['covar'][k] )
    else:
      for k in xrange( self.K):
  	    postDistr = self.obsPrior.getPosteriorDistr( SS['N'][k], SS['mean'][k], SS['covar'][k] )
  	    self.qobsDistr[k].rho_update( postDistr, rho )

  def get_obs_suff_stats( self, SS, Data, LP ):
    ''' Suff Stats
    '''
    if type(Data) is dict:
      X = Data['X']
    else:
      X = Data
    resp = LP['resp']

    SS['mean'] = np.dot( resp.T, X ) / SS['N'][:, np.newaxis]
    SS['covar'] = np.empty( (self.K, self.D, self.D) )
    for k in xrange( self.K):
      Xdiff = X - SS['mean'][k]
      SS['covar'][k] = np.dot( Xdiff.T * resp[:,k], Xdiff )
      SS['covar'][k] /= SS['N'][k]
    return SS
    
  def E_log_soft_ev_mat( self, X ):
    ''' E-step update
    '''
    N,D = X.shape
    lpr = np.empty( (X.shape[0], self.K) )
    for k in xrange( self.K ):
      lpr[:,k] = self.qobsDistr[k].E_log_pdf( X )
    return lpr
    
  def E_logpX( self, LP, SS ):
    ''' E_{q(Z), q(Phi)} [ log p(X) ]
       Bishop PRML eq. 10.71
    '''
    lpX = -self.D*LOGTWOPI*np.ones( self.K )
    dist = np.zeros( self.K)
    for k in range( self.K ):
      lpX[k] += self.qobsDistr[k].ElogdetLam - self.D/self.qobsDistr[k].kappa \
                - self.qobsDistr[k].dF* self.qobsDistr[k].traceW( SS['covar'][k] )  \
                - self.qobsDistr[k].dF* self.qobsDistr[k].dist_mahalanobis( SS['mean'][k] )
    return 0.5*np.inner(SS['N'],lpX)
    
  def E_logpPhi( self ):
    return self.E_logpLam() + self.E_logpMu()
      
  def E_logqPhi( self ):
    return self.E_logqLam() + self.E_logqMu()
  
  def E_logpMu( self ):
    ''' First four RHS terms (inside sum over K) in Bishop 10.74
    '''
    lp = np.empty( self.K)    
    for k in range( self.K ):
      mWm = self.qobsDistr[k].dist_mahalanobis( self.obsPrior.m )
      lp[k] = self.qobsDistr[k].ElogdetLam \
                -self.D*self.obsPrior.kappa/self.qobsDistr[k].kappa \
                -self.obsPrior.kappa*self.qobsDistr[k].dF*mWm
    lp += self.D*( np.log( self.obsPrior.kappa ) - LOGTWOPI)
    return 0.5*lp.sum()
    
  def E_logpLam( self ):
    ''' Last three RHS terms in Bishop 10.74
    '''
    lp = np.empty( self.K) 
    for k in xrange( self.K ):
      lp[k] = 0.5*(self.obsPrior.dF - self.D -1)*self.qobsDistr[k].ElogdetLam
      lp[k] -= 0.5*self.qobsDistr[k].dF*self.qobsDistr[k].traceW(self.obsPrior.invW)
    return lp.sum() + self.K * self.obsPrior.logWishNormConst() 
    
  def E_logqMu( self ):
    ''' First two RHS terms in Bishop 10.77
    '''
    
    lp = np.zeros( self.K)
    for k in xrange( self.K):
      lp[k] = 0.5*self.qobsDistr[k].ElogdetLam \
              + 0.5*self.D*( np.log( self.qobsDistr[k].kappa ) - LOGTWOPI )
    return lp.sum() - 0.5*self.D*self.K
                     
  def E_logqLam( self ):
    ''' Last two RHS terms in Bishop 10.77
    '''
    lp = np.zeros( self.K)
    for k in xrange( self.K):
      lp[k] -= self.qobsDistr[k].entropyWish()
    return lp.sum()
