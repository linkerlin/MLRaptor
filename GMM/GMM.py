"""
Gaussian Mixture Model
  for real-valued numeric data of any dimension
  supporting multiple types of covariance structures
    (diagonal or full)  

Author:  Mike Hughes (michaelchughes.com)

Usage
-------
mygmm = GMM( K=1, covar_type='diag' )

Params
-------
  w     : Kx1 mixture weights 
            w[k] = probability of choosing component k.
            sum(w) must equal 1.
  mu    : KxD 
            mu[k] = vector of location of k-th cluster
  Sigma : K x variable_size
            Sigma[k] = covariance params for k-th cluster
          

Inference
-------
See EMLearnerGMM.py for EM algorithm
    VBLearnerGMM.py for Variation Bayes algorithm

References
-------
Pattern Recognition and Machine Learning by C. Bishop

"""

import numpy as np
from MLUtil import logsumexp

def np2flatstr( X, fmt='% .3f' ):
  return ' '.join( [fmt % x for x in X.flatten() ] )  

class GMM(object):
  '''
    Describes essential properties of the Gaussian mixture model,
      a generative model for multivariate real-valued data    
              
  '''
  def __str__( self ):
    s = '%d component Gaussian mixture model\n' % (self.K)
    if self.w is None:
      s += ' not yet initialized.\n'
      return s
    s += 'w=%s\n' % ( np2flatstr( self.w, '%.2f' ) )
    for k in range( np.minimum(10,self.K) ):
      s += 'mu[%d]= %s\n' % ( k, np2flatstr( self.mu[k], '% 5.1f' ) )
    for k in range( np.minimum(10,self.K) ):
      s += 'sigma[%d]= %s\n' % ( k, np2flatstr( self.Sigma[k], '% 5.1f' ) )
    return s
  
  def __init__( self, K=1, covar_type='diag'):
    self.K = K
    self.covar_type = covar_type
    self.D = None
    
    self.w  = None
    self.mu = None
    self.Sigma = None
    
    self.mu_prior = None
    self.sigma_prior = None
    self.w_prior = None
    
  def add_w_prior( self, name, alpha=None):
    self.w_prior = name
    if name == 'Dir':
      self.alpha = alpha
      
  def add_mu_prior( self, name, mean=None, covar=None):
    self.mu_prior = name
    if name == 'Normal':
      self.muP_mean = mean
      self.muP_covar  = covar
       
  def add_sigma_prior(self, name, degFree=None, smat=None):
    self.sigma_prior = name
    if name == 'InvWishart':
      self.sigP_degFree = degFree
      self.sigP_smat = smat

  def calc_posterior_prob_mat(self, X):
    """Compute posterior probability for hidden component assignment Z
          under current model parameters mu,sigma given data X
       Parameters:
         X : array_like, N x D

       Returns:  
         resp : N x K
              =  Pr( z_n = k | x_n, \mu_k, \Sigma_k )
                  see Bishop PRML eq. 9.23 (p. 438)
    """
    lpr = np.log( self.w ) + self.calc_soft_evidence_mat( X )
    lprSUM = logsumexp(lpr, axis=1)
    return np.exp(lpr - lprSUM[:, np.newaxis])
    
  def calc_evidence(self, X):
    """Compute evidence for given data X under current model
       Parameters:
         X : array_like, N x D

       Returns:  
         evBound : scalar real
              =  \sum_n log( \sum_k w_k * N( x_n | \mu_k, \sigma_k )
                  see Bishop PRML eq. 9.28 (p. 439)
    """
    lpr = np.log( self.w ) + self.calc_soft_evidence_mat( X )
    evidenceBound = logsumexp(lpr, axis=1).sum()
    return evidenceBound

  def calc_soft_evidence_mat(self, X):
    """Compute Gaussian log-density at X for a diagonal model
       Parameters:
         X : array_like, N x D

       Returns:  
         lpr : NxK matrix
               where lpr[n,k] = Pr( X[n,:] | mu[k,:], sigma[k,:] )
    """
    N, D = X.shape
    Sigma = self.Sigma
    Mu    = self.mu
    lpr = np.dot( X**2, (1.0/Sigma).T ) # x^T invS x term
    lpr -= 2*np.dot( X, (Mu/Sigma).T )  # x^T invS mu term
    lpr += np.sum( Mu**2/Sigma, axis=1) # mu^T invS mu term
    lpr += D * np.log(2*np.pi)  # norm constants
    lpr += np.sum(np.log(Sigma), axis=1)
    return -0.5*lpr
    
  def generate_data(self, N, D):
    assert D == self.mu.shape[1]
    assert D == self.Sigma.shape[1]
    Nk = np.random.mtrand.multinomial( N, self.w )
    X = np.zeros( (0,D) )
    for k in range(self.K):
      Xk = self.mu[k] + self.Sigma[k]*np.random.randn( Nk[k], D )
      X = np.vstack( [X, Xk] )
    return X

