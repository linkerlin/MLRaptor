'''
 Factorized Variational Distribution
   for approximating a Gaussian Admixture Model (GAM)

 W_g := mixture weights for group "g"

 q( Z, W, mu, Sigma ) = q(Z) \prod_{g=1}^G q(W_g) q(mu|Sigma)q(Sigma)

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

from MLUtil import logsumexp
import GMM

EPS = 1e-13
LOGPI = np.log(np.pi)
LOGTWO = np.log(2.00)
LOGTWOPI = np.log( 2.0*np.pi )

class QGAM( object ):

  def __init__(self, obsPrior, K=2, alpha0=1.0, **kwargs):
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
    self.logkappa     = np.log( np.asarray( [q.kappa for q in self.qObs] ) )


  def E_step(self, Data, LocalP ):
    GroupIDs = Data['GroupIDs']
    X = Data['X']
    N,D = X.shape
    assert self.D == D

    # Create lpr : N x K matrix
    #   where lpr[n,k] =def= log r[n,k], as in Bishop PRML eq 10.67
    lpr = np.empty( (N, self.K) )

    # LIKELIHOOD Terms
    for k in range(self.K):
      lpr[:,k] = -0.5*self.qObs[k].dF*self.qObs[k].dist_mahalanobis( X ) \
                 -0.5*D/self.qObs[k].kappa
    lpr += 0.5*self.logdetLam

    # PRIOR Terms
    for gg in xrange( len(GroupIDs) ):
      lpr[ GroupIDs[gg] ] += LocalP['Elogw_perGroup'][gg]


    lprSUM = logsumexp(lpr, axis=1)
    resp   = np.exp(lpr - lprSUM[:, np.newaxis])
    resp   /= resp.sum( axis=1)[:,np.newaxis] # row normalize

    return resp

  def M_step( self, SS ):
    '''M-step of the EM alg.
         for updates to w, mu, and Sigma
    '''
    alpha_perGroup = self.alpha0 + SS['NperGroup']
    logw_perGroup  = digamma( alpha_perGroup ) - digamma( alpha_perGroup.sum(axis=1) )[:,np.newaxis]

    for k in xrange( self.K ):
      self.qObs[k] = self.prior.getPosteriorParams( \
                        SS['N'][k], SS['mean'][k], SS['covar'][k] )
    self.update_helper_params()

    LocalP = dict()
    LocalP['alpha_perGroup']  = alpha_perGroup
    LocalP['Elogw_perGroup']   = logw_perGroup

    return LocalP
      
  def calc_suff_stats(self, Data, resp):
    GroupIDs = Data['GroupIDs']
    X = Data['X']
    SS = dict()
    SS['N'] = np.sum( resp, axis=0 ) + EPS # add small pos. value to avoid nan
    SS['NperGroup'] = np.zeros( (len(GroupIDs),self.K)  )
    for gg in range( len(GroupIDs) ):
      SS['NperGroup'][gg] = np.sum( resp[ GroupIDs[gg] ], axis=0 )
    SS['mean'] = np.dot( resp.T, X ) / SS['N'][:, np.newaxis]
    SS['covar'] = np.zeros( (self.K, self.D, self.D) )
    for k in range( self.K):
      Xdiff = X - SS['mean'][k]
      SS['covar'][k] = np.dot( Xdiff.T, Xdiff * resp[:,k][:,np.newaxis] )
      SS['covar'][k] /= SS['N'][k]
    return SS

  def calc_ELBO( self, Data, resp, SS, LocalP ):
    GroupIDs = Data['GroupIDs']
    ELBO = self.ElogpX( resp, SS) \
           +self.ElogpZ( GroupIDs, resp, LocalP ) - self.ElogqZ( GroupIDs, resp ) \
           +self.ElogpW(  LocalP)      - self.ElogqW( LocalP)       \
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
                - self.qObs[k].dF* self.qObs[k].dist_mahalanobis( SS['mean'][k] )
    return 0.5*np.inner(SS['N'],lpX)
    
  def ElogpZ( self, GroupIDs, resp, LocalP ):
    ElogpZ = 0
    for gg in xrange( len(GroupIDs) ):
      ElogpZ += np.sum( resp[GroupIDs[gg]] * LocalP['Elogw_perGroup'][gg] )
    return ElogpZ
    
  def ElogqZ( self, GroupIDs, resp ):
    ElogqZ = 0
    for gg in xrange( len(GroupIDs) ):
      ElogqZ += np.sum( resp[GroupIDs[gg]] * resp[GroupIDs[gg]] )
    return  ElogqZ
    
  def ElogpW( self, LP ):
    nGroup = len(LP['alpha_perGroup'])
    ElogpW = gammaln(self.K*self.alpha0)-self.K*gammaln(self.alpha0)    
    ElogpW *= nGroup  # same prior over each group of data!
    for gg in xrange( nGroup ):
      ElogpW += (self.alpha0-1)*LP['Elogw_perGroup'][gg].sum()
    return ElogpW
 
  def ElogqW( self, LP ):
    ElogqW = 0
    for gg in xrange( len(LP['alpha_perGroup']) ):
      a_gg = LP['alpha_perGroup'][gg]
      ElogqW +=  gammaln(  a_gg.sum()) - gammaln(  a_gg ).sum() \
                  + np.inner(  a_gg -1,  LP['Elogw_perGroup'][gg] )
    return ElogqW
             
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

  #########################################################  OLD WAY
  def update_local( self, X, GroupIDs ):
    LP = dict()

    return LP

  def get_global_suff_stats( self, X, GroupIDs, LP ):
    SS = dict()
    return SS

  def update_global( self, SS ):
    pass
