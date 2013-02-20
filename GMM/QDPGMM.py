'''
 Factorized Variational Distribution
   for approximating a Dirichlet-Process Gaussian Mixture Model
   using the stick-breaking construction and truncating to at most K components

 Author: Mike Hughes (mike@michaelchughes.com)

 Model
 -------
 for each component k=1...K:
    stick length  v_k ~ Beta( 1, alpha0 )
    mixture weight  w_k <-- v_k * \prod_{l < k}(1-v_l)

 Variational Approximation
 -------
 q( Z, v, mu, Sigma ) = q(Z)q(v)q(mu|Sigma)q(Sigma)

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

from MLUtil import logsumexp
import GMM

EPS = 1e-13
LOGPI = np.log(np.pi)
LOGTWO = np.log(2.00)
LOGTWOPI = np.log( 2.0*np.pi )

class QDPGMM( object ):

  def __init__(self, obsPrior, K=2, alpha0=1.0, **kwargs):
    self.K = K
    self.alpha1 = 1.0
    self.alpha0 = alpha0

    self.prior = obsPrior
    self.D     = obsPrior.D

    #  q( v_k ) = Beta( qalpha1[k], qalpha0[k] )
    self.qalpha1 = np.zeros( K )
    self.qalpha0 = np.zeros( K )

    self.qObs  = [obsPrior for k in range(K)]
  
  def to_alpha_string( self ):
    alphs = np.concatenate( [self.qalpha0, self.qalpha1] )
    return ' '.join( ['%.5f'%(x) for x in alphs] )
    
  def to_obs_string( self, k):
    return str( self.qObs[k] )
  
  def update_helper_params(self):
    '''
      E[ log p( Z | V ) ] = \sum_n E[ log p( Z[n] | V )
         = \sum_n E[ log p( Z[n]=k | w(V) ) ]
         = \sum_n \sum_k z_nk log w(V)_k
      where log w(V)_k = log[  V_k \prod{l<k} (1 - V_l ) ]
    '''
    self.logdetLam = np.asarray( [q.ElogdetLam() for q in self.qObs] )
    self.logkappa     = np.log( np.asarray( [q.kappa for q in self.qObs] ) )

    DENOM = digamma( self.qalpha0 + self.qalpha1 )
    self.ElogV      = digamma( self.qalpha1 ) - DENOM
    self.Elog1mV    = digamma( self.qalpha0 ) - DENOM

    self.Elogw = self.ElogV.copy()
    self.Elogw[1:] += self.Elog1mV.cumsum()[:-1]
    
  def E_step(self, X):
    N,D = X.shape
    assert self.D == D
    # Create lpr : N x K matrix
    #   where lpr[n,k] =def= log r[n,k], as in Bishop PRML eq 10.67
    lpr = np.empty( (N, self.K) )
    for k in range(self.K):
      # calculate the ( x_n - m )'*W*(x_n-m) term
      lpr[:,k] = -0.5*self.qObs[k].dF*self.qObs[k].dist_mahalanobis( X ) \
                 -0.5*D/self.qObs[k].kappa
    lpr += self.Elogw
    lpr += 0.5*self.logdetLam
    lprSUM = logsumexp(lpr, axis=1)
    resp   = np.exp(lpr - lprSUM[:, np.newaxis])
    resp   /= resp.sum( axis=1)[:,np.newaxis] # row normalize
    return resp

  def M_step( self, SS ):
    '''M-step of the EM alg.
         for updates to w, mu, and Sigma
    '''
    self.qalpha1 = self.alpha1 + SS['N']
    self.qalpha0[:-1] = self.alpha0 + SS['N'][::-1].cumsum()[::-1][1:]
    self.qalpha0[-1]  = self.alpha0
    for k in xrange( self.K ):
      self.qObs[k] = self.prior.getPosteriorParams( \
                        SS['N'][k], SS['mean'][k], SS['covar'][k] )
    self.update_helper_params()

  def calc_suff_stats(self, X, resp):
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
           +self.ElogpV( )      - self.ElogqV()       \
           +self.ElogpMu()      - self.ElogqMu()      \
           +self.ElogpLam()     - self.ElogqLam() 
    return ELBO 

  def ElogpX( self, resp, SS ):
    ''' Bishop PRML eq. 10.71
    '''
    lpX = -self.D*LOGTWOPI*np.ones( self.K )
    for k in range( self.K ):
      lpX[k] += self.qObs[k].ElogdetLam() - self.D/self.qObs[k].kappa \
                - self.qObs[k].dF* self.qObs[k].traceW( SS['covar'][k] )  \
                - self.qObs[k].dF* self.qObs[k].dist_mahalanobis( SS['mean'][k])
    return 0.5*np.inner(SS['N'],lpX)
    
  def ElogpZ( self, resp ):
    '''
      E[ log p( Z | V ) ] = \sum_n E[ log p( Z[n] | V )
         = \sum_n E[ log p( Z[n]=k | w(V) ) ]
         = \sum_n \sum_k z_nk log w(V)_k
    '''
    return np.sum( resp * self.Elogw ) 
    
  def ElogqZ( self, resp ):
    return np.sum( resp *np.log(resp+EPS) )
    

  ############################################################## stickbreak terms
  def ElogpV( self ):
    '''
      E[ log p( V | alpha ) ] = sum_{k=1}^K  E[log[   Z(alpha) Vk^(a1-1) * (1-Vk)^(a0-1)  ]]
         = sum_{k=1}^K log Z(alpha)  + (a1-1) E[ logV ] + (a0-1) E[ log (1-V) ]
    '''
    logZprior = gammaln( self.alpha0 + self.alpha1 ) - gammaln(self.alpha0) - gammaln( self.alpha1 )
    logEterms  = (self.alpha1-1)*self.ElogV + (self.alpha0-1)*self.Elog1mV
    return self.K*logZprior + logEterms.sum()    

  def ElogqV( self ):
    '''
      E[ log q( V | qa ) ] = sum_{k=1}^K  E[log[ Z(qa) Vk^(ak1-1) * (1-Vk)^(ak0-1)  ]]
       = sum_{k=1}^K log Z(qa)   + (ak1-1) E[logV]  + (a0-1) E[ log(1-V) ]
    '''
    logZq = gammaln( self.qalpha0 + self.qalpha1 ) - gammaln(self.qalpha0) - gammaln( self.qalpha1 )
    logEterms  = (self.qalpha1-1)*self.ElogV + (self.qalpha0-1)*self.Elog1mV
    return logZq.sum() + logEterms.sum()    


  ############################################################## Mu, Lam terms
             
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
