'''
 Factorized Variational Distribution
   for approximating a Gaussian Mixture Model

 q( Z, w, mu, Sigma ) = q(Z)q(w)q(mu|Sigma)q(Sigma)

 Author: Mike Hughes (mike@michaelchughes.com)

 Parameters
 -------
   alpha0   : scalar hyperparameter of symmetric Dirichlet prior on mix. weights
   obsPrior : Python object that represents prior on emission params (mu,Sigma)
                  conventionally, obsPrior = Gaussian-Wishart distribrution 
                    see GaussWishDistr.py
   
 Usage
 -------
   gw   = GaussWishDistr( dF=3, invW=np.eye(2)  )
   qgmm = QGMM( K=10, alpha0=0.1, obsPrior=gw )

 Inference
 -------
   See VBLearnerGMM.py

 References
 -------
   Pattern Recognition and Machine Learning, by C. Bishop.
'''

import numpy as np
from scipy.special import digamma, gammaln
from sklearn.utils.extmath import logsumexp
import GMM

EPS = 1e-13
LOGPI = np.log(np.pi)
LOGTWO = np.log(2.00)
LOGTWOPI = np.log( 2.0*np.pi )

class QGMM( object ):

  def __init__(self, K, alpha0, obsPrior):
    self.K = K
    self.alpha0 = alpha0
    self.prior = obsPrior
    self.D     = obsPrior.D
    self.alpha = np.zeros( K )
    self.qObs  = [obsPrior for k in range(K)]
  
  def to_alpha_string( self ):
    return ' '.join( ['%.5f'%(x) for x in self.alpha] )
    
  def to_obs_string( self, k):
    return str( self.qObs[k] )
  
  def update_helper_params(self):
    self.logdetLam = np.asarray( [q.ElogdetLam() for q in self.qObs] )
    self.logw      = digamma( self.alpha ) - digamma( self.alpha.sum() )
    self.logkappa     = np.log( np.asarray( [q.kappa for q in self.qObs] ) )
    
  def E_step(self, X):
    N,D = X.shape
    assert self.D == D
    # Create lpr : N x K matrix
    #   where lpr[n,k] =def= log r[n,k], as in Bishop PRML eq 10.67
    lpr = np.zeros( (N, self.K) )
    for k in range(self.K):
      # calculate the ( x_n - m )'*W*(x_n-m) term
      lpr[:,k] = -0.5*self.qObs[k].dF*self.qObs[k].dist_mahalanobis( X ) \
                 -0.5*D/self.qObs[k].kappa
    lpr += self.logw
    lpr += 0.5*self.logdetLam
    lprSUM = logsumexp(lpr, axis=1)
    resp   = np.exp(lpr - lprSUM[:, np.newaxis])
    resp   /= resp.sum( axis=1)[:,np.newaxis] # row normalize
    return resp

  def M_step( self, SS ):
    '''M-step of the EM alg.
         for updates to w, mu, and Sigma
    '''
    self.alpha   = self.alpha0 + SS['N']
    for k in xrange( self.K ):
      self.qObs[k] = self.prior.getPosteriorParams( \
                        SS['N'][k], SS['mean'][k], SS['covar'][k] )
    self.update_helper_params()  
      
  def calc_suff_stats(self, X, resp):
    SS = dict()
    SS = dict()
    SS['N'] = np.sum( resp, axis=0 ) + EPS # add small pos. value to avoid nan
    SS['mean'] = np.dot( resp.T, X ) / SS['N'][:, np.newaxis]
    SS['covar'] = np.zeros( (self.K, self.D, self.D) )
    for k in range( self.K):
      Xdiff = X - SS['mean'][k]
      SS['covar'][k] = np.dot( Xdiff.T, Xdiff * resp[:,k][:,np.newaxis] )
      SS['covar'][k] /= SS['N'][k]
    return SS    
    
  def calc_ELBO( self, resp=None, SS=None, X=None):
    if SS is None or resp is None:
      if X is None: raise ArgumentError('Need data to compute bound for')
      resp = self.E_step( X )
      SS = self.calc_suff_stats( X, resp)
    ELBO = self.ElogpX( resp, SS) \
           +self.ElogpZ( resp ) - self.ElogqZ( resp ) \
           +self.ElogpW( )      - self.ElogqW()       \
           +self.ElogpMu()      - self.ElogqMu()      \
           +self.ElogpLam()     - self.ElogqLam() 
    return ELBO 

  def ElogpX( self, resp, SS ):
    ''' Bishop PRML eq. 10.71
    '''
    lpX = -self.D*LOGTWOPI*np.ones( self.K )
    dist = np.zeros( self.K)
    for k in range( self.K ):
      lpX[k] += self.qObs[k].ElogdetLam() - self.D/self.qObs[k].kappa \
                - self.qObs[k].dF* self.qObs[k].traceW( SS['covar'][k] )  \
                - self.qObs[k].dF* self.qObs[k].dist_mahalanobis( SS['mean'][k])
      dist[k] = self.qObs[k].dist_mahalanobis( SS['mean'][k])
    return 0.5*np.inner(SS['N'],lpX)
    
  def ElogpZ( self, resp ):
    ''' Bishop PRML eq. 10.72
    '''
    return np.sum( resp * self.logw )
    
  def ElogqZ( self, resp ):
    ''' Bishop PRML eq. 10.75
    '''
    return np.sum( resp *np.log(resp+EPS) )
    
  def ElogpW( self ):
    ''' Bishop PRML eq. 10.73
    '''
    return gammaln(self.K*self.alpha0)-self.K*gammaln(self.alpha0) \
             + (self.alpha0-1)*self.logw.sum()
 
  def ElogqW( self ):
    ''' Bishop PRML eq. 10.76
    '''
    return gammaln(self.alpha.sum())-gammaln(self.alpha).sum() \
             + np.inner( (self.alpha-1), self.logw )
             
  def ElogpMu( self ):
    ''' First four RHS terms (inside sum over K) in Bishop 10.74
    '''
    lp = self.logdetLam \
         + self.D*( np.log( self.prior.kappa ) - LOGTWOPI)
    for k in range( self.K ):
      mWm = self.qObs[k].dist_mahalanobis( self.prior.m )
      lp[k] +=  -self.D*self.prior.kappa/self.qObs[k].kappa \
                -self.prior.kappa*self.qObs[k].dF* mWm
    return 0.5*lp.sum()
    
  def ElogpLam( self ):
    ''' Last three RHS terms in Bishop 10.74
    '''
    lp = 0.5*(self.prior.dF - self.D -1)*self.logdetLam
    for k in xrange( self.K ):
      lp[k] -= 0.5*self.qObs[k].dF*self.qObs[k].traceW(self.prior.invW)
    return lp.sum() + self.K * self.prior.logWishNormConst() 
    
  def ElogqMu( self ):
    ''' First two RHS terms in Bishop 10.77
    '''
    lp =  0.5*self.logdetLam + 0.5*self.D*( self.logkappa - LOGTWOPI ) \
          -0.5*self.D
    return lp.sum()
                     
  def ElogqLam( self ):
    ''' Last two RHS terms in Bishop 10.77
    '''
    lp = np.zeros( self.K)
    for k in xrange( self.K):
      lp[k] -= self.qObs[k].entropyWish()
    return lp.sum()
    
  #############################################  estimate GMM params    
  def estGMM_MAP( self ):
    w = self.alpha - 1
    assert np.all( w > 0 )
    w /= w.sum()
    
    mu = np.zeros( (self.K, self.D) )
    Sigma = np.zeros( (self.K, self.D,  self.D) )
    
    for k in xrange(self.K):
      m,S = self.qObs[k].getMAP()
      mu[k] = m
      Sigma[k] = S
    mygmm = GMM.GMM( self.K, covariance_type='full')
    mygmm.w = w
    mygmm.mu = mu
    mygmm.Sigma = Sigma
    return mygmm
  
  def estGMM_Mean( self ):
    w = self.alpha
    w /= w.sum()
    
    mu = np.zeros( (self.K, self.D) )
    Sigma = np.zeros( (self.K, self.D,  self.D) )
    
    for k in xrange(self.K):
      m, S = self.qObs[k].getMean()
      mu[k] = m
      Sigma[k] = S
    mygmm = GMM.GMM( self.K, covariance_type='full')
    mygmm.w = w
    mygmm.mu = mu
    mygmm.Sigma = Sigma
    return mygmm
