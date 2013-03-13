#! /usr/bin/python -W ignore::DeprecationWarning
"""
Variational Bayesian learning algorithm
  for a Gaussian Mixture Model (GMM)
"""
import os.path
import time
import numpy as np
import sklearn.cluster
from sklearn.utils.extmath import logsumexp
from scipy.special import digamma, gammaln, betaln
import scipy.linalg

EPS = np.finfo(float).eps
TWOPI = np.pi*2.0

def np2flatstr( X, fmt='% .6f' ):
  return ' '.join( [fmt % x for x in X.flatten() ] )  

def logNormConstWish( W, dF ):
  D = W.shape[0]
  lC = -0.5*dF*np.log( np.linalg.det(W) )
  lC -= 0.5*dF*D* np.log(2)
  lC -= D*(D-1)/4*np.log(np.pi)
  lC -= np.sum( gammaln( 0.5*(dF+1- np.arange(1,D+1)) ) )
  return lC

class VBLearnerGMM(object):

  def __init__( self, gmm, alpha0, obsPrior, savefilename='GMMtrace', \
                      initname='kmeans', Niter=10, printEvery=25, saveEvery=5 ):
    self.gmm = gmm
    self.obsPrior = obsPrior
    
    self.alpha0 = alpha0
    self.qMixComp  = [None for k in range(gmm.K)]
        
    self.savefilename = savefilename
    self.initname = initname
    self.Niter = Niter
    self.printEvery = printEvery
    self.saveEvery = saveEvery
    
  def getEstGMM( self, type='mean' ):
    if type=='mean':
      return self.getEstGMM_Mean()
    elif type=='mode':
      return self.getEstGMM_MAP()
      
  def getEstGMM_MAP( self ):
    w = self.alpha - 1
    assert np.all( w > 0 )
    self.gmm.w = w/w.sum()
    
    self.gmm.mu = np.zeros( (self.gmm.K, self.gmm.D) )
    self.gmm.Sigma = np.zeros( (self.gmm.K, self.gmm.D,  self.gmm.D) )
    
    for k in xrange(self.gmm.K):
      mu,Sigma = self.qMixComp[k].getMAP()
      self.gmm.mu[k] = mu
      self.gmm.Sigma[k] = Sigma
    return self.gmm  
  
  def getEstGMM_Mean( self ):
    self.gmm.w = self.alpha
    self.gmm.w /= self.gmm.w.sum()
    
    self.gmm.mu = np.zeros( (self.gmm.K, self.gmm.D) )
    self.gmm.Sigma = np.zeros( (self.gmm.K, self.gmm.D,  self.gmm.D) )
    
    for k in xrange(self.gmm.K):
      mu,Sigma = self.qMixComp[k].getMean()
      self.gmm.mu[k] = mu
      self.gmm.Sigma[k] = Sigma
    return self.gmm
    
  def init_params( self, X, seed): 
    np.random.seed( seed ) 
    if self.initname == 'kmeans':
      self.init_kmeans(X, seed)
      
  def init_kmeans( self, X, seed):
    N,D = X.shape
    K = self.gmm.K
    self.gmm.D = D
    km = sklearn.cluster.KMeans( K, random_state=seed )
    km.fit(X)
    grid = np.tile( np.arange(K), (N,1) )
    resp = np.asfarray( grid == km.labels_[:,np.newaxis] ) # make NxK matrix of hard cluster asgn
    SS = self.calc_suff_stats( X, resp)
    
    self.M_step( SS )
    self.alpha = sum(self.alpha)/self.gmm.K* np.ones( self.gmm.K ) 
      
  def fit( self, X, seed=None, convTHR=1e-4):
    self.start_time = time.time()
    prevBound = -np.inf
    status = 'max iters reached.'
    for iterid in xrange(self.Niter):
      if iterid==0:
        self.init_params( X, seed )
        resp = self.E_step( X )
        SS = self.calc_suff_stats( X, resp )
      else:
        self.M_step( SS )
        resp = self.E_step( X )
        SS = self.calc_suff_stats( X, resp )
      evBound = self.calc_ELBO( resp, SS )
      self.save_state(iterid, evBound)
      self.print_state(iterid, evBound)
      assert prevBound <= evBound
      if iterid > 3 and np.abs( evBound-prevBound )/np.abs(evBound) <= convTHR:
        status = 'converged.'
        break
      prevBound = evBound
    #Finally, save, print and exit 
    self.save_state(iterid, evBound) 
    self.print_state(iterid, evBound, doFinal=True, status=status)
  
  def calc_suff_stats_long( self, X, resp ):
    D = X.shape[1]
    N = np.sum( resp,axis=0)
    M = np.dot( resp.T, X ) / N[:,np.newaxis]
    C = np.zeros( (self.gmm.K,D,D) )
    for k in range( self.gmm.K ):
      Q = np.zeros( (D,D) )
      for n in range( X.shape[0] ):
        Xd = X[n] - M[k]
        Q += resp[n,k] * np.outer( Xd, Xd )
      C[k] = Q/N[k]
    return N,M,C
  
  def calc_suff_stats( self, X, resp ):
    SS = dict()
    SS['N'] = np.sum( resp, axis=0 )
    SS['mean'] = np.dot( resp.T, X ) / SS['N'][:, np.newaxis]
    SS['covar'] = np.zeros( (self.gmm.K, self.gmm.D, self.gmm.D) )
    for k in range( self.gmm.K):
      Xdiff = X - SS['mean'][k]
      SS['covar'][k] = np.dot( Xdiff.T, Xdiff * resp[:,k][:,np.newaxis] )
      SS['covar'][k] /= SS['N'][k]
    return SS
    
  def E_step( self, X):
    N,D = X.shape
    lpr = np.zeros( (N, self.gmm.K) )
    logdet = np.zeros( self.gmm.K )
    dterms = np.arange( 1,D+1 ) # 1,2,3... D
    self.invWchol = list()
    for k in range(self.gmm.K):
      dXm  = X - self.qMixComp[k].m
      L = scipy.linalg.cholesky(  self.qMixComp[k].invW, lower=True)
      self.invWchol.append( L )
      
      if np.any( np.isnan(L) | np.isinf(L) ):
        print 'NaN!', self.qMixComp[k]
      #invL = scipy.linalg.inv( L )
      #  want: Q =  invL * X.T
      #    so we solve for matrix Q s.t. L*Q = X.T
      lpr[:,k] = -0.5*self.qMixComp[k].dF \
                    * np.sum( scipy.linalg.solve_triangular( L, dXm.T,lower=True)**2, axis=0)
      lpr[:,k] -= 0.5*D/self.qMixComp[k].beta
      # det( W ) = 1/det(invW)
      #          = 1/det( L )**2 
      # det of triangle matrix = prod of diag entries
      logdet[k] = -2*np.sum( np.log(np.diag(L) ) ) + D*np.log(2.0) 
      logdet[k] += digamma( 0.5*(dterms+1+self.qMixComp[k].dF)  ).sum()
    self.logwtilde = digamma( self.alpha ) - digamma( self.alpha.sum() )
    self.logLtilde = logdet
    lpr += self.logwtilde
    lpr += logdet
    lprSUM = logsumexp(lpr, axis=1)
    resp   = np.exp(lpr - lprSUM[:, np.newaxis])
    resp   /= resp.sum( axis=1)[:,np.newaxis] # row normalize
    return resp
    
  def M_step( self, SS ):
    '''M-step of the EM alg.
       Updates internal mixture model parameters
         to maximize the evidence of given data X  (aka probability of X)
       See Bishop PRML Ch.9 eqns 9.24, 9.25, and 9.26
         for updates to w, mu, and Sigma
    '''
    self.alpha   = self.alpha0 + SS['N']
    
    for k in xrange( self.gmm.K ):
      self.qMixComp[k] = self.obsPrior.getPosteriorParams( SS['N'][k], SS['mean'][k], SS['covar'][k] )
  
  def dist_mh( self, x, invSigchol ):
    try:
      return (scipy.linalg.solve_triangular( invSigchol, x, lower=True)**2 ).sum()    
    except Exception:
      print 'OH NO!'
      print invSigchol
      raise
      
  def calc_ELBO( self, resp, SS ):
    D = self.gmm.D
    K = self.gmm.K
    try:
      logw = self.logwtilde
    except Exception:
      logw = digamma(self.alpha) - digamma(self.alpha.sum() )
      
    try:
      logL = self.logLtilde
    except Exception:
      dterms = np.arange( 1,D+1 )
      self.logLtilde = [None for k in range( self.gmm.K)]
      self.invWchol = list()
      self.detW = np.zeros( self.gmm.K)
      for k in range( self.gmm.K ):
        L = scipy.linalg.cholesky( self.qMixComp[k].invW, lower=True )
        self.invWchol.append( L )
        self.detW[k] = -2*np.sum( np.log(np.diag(L) ) )
        self.logLtilde[k] = digamma( 0.5*(self.qMixComp[k].dF+1-dterms)  ).sum() 
        self.logLtilde[k]+= self.detW[k] + D*np.log(2.0) 
        
        
    alpha0 = self.alpha0 * np.ones( self.gmm.K)
    Ep_w = np.inner(     alpha0-1, logw ) \
              + gammaln(  alpha0.sum() ) - gammaln( alpha0).sum()
    Eq_w = np.inner( self.alpha-1, logw ) \
              + gammaln( self.alpha.sum() ) - gammaln(self.alpha).sum()
    Ep_z  = ( resp * logw ).sum()
    Eq_z  = ( resp * np.log(resp+EPS) ).sum()
    
    print 'Ep[z] %.4e | Eq[z] %.4e' % (Ep_z , Eq_z )
    print 'Ep[w] %.4e | Eq[w] %.4e' % (Ep_w , Eq_w )
    
    Ep_X = 0
    exk  = np.zeros( 4 )
    Ep_mL = K *logNormConstWish( np.linalg.pinv(self.obsPrior.invW), self.obsPrior.dF )
    Ep_mL += 0.5*(self.obsPrior.dF-D-1)*np.sum( self.logLtilde )
    
    Eq_mL = -0.5*K*D
    xWx = list()
    SWtr = list()
    for k in range( self.gmm.K ):
      dFk = self.qMixComp[k].dF 
      Wk = np.linalg.inv( self.qMixComp[k].invW )
      dXm = SS['mean'][k] - self.qMixComp[k].m
      #Dist = np.sum(scipy.linalg.solve_triangular(self.invWchol[k],dXm,lower=True)**2)
      Dist  = self.dist_mh( dXm, self.invWchol[k] )
      Ep_Xk  = self.logLtilde[k] - D/self.qMixComp[k].beta 
      Ep_Xk -= dFk *np.trace( np.dot(SS['covar'][k], Wk ) )  
      Ep_Xk -= dFk *Dist 
      Ep_Xk -= D*np.log(TWOPI)
      xWx.append( Dist )
      SWtr.append( np.trace( np.dot(SS['covar'][k], Wk ) ) )
      
      mDist = self.dist_mh( self.qMixComp[k].m-self.obsPrior.m, self.invWchol[k])
      Ep_muk  = D*np.log(self.obsPrior.beta/TWOPI) + self.logLtilde[k]
      Ep_muk -= D*self.obsPrior.beta/self.qMixComp[k].beta
      Ep_muk -= self.obsPrior.beta* dFk *mDist
      
      Ep_mLam -= dFk*np.trace( np.dot(self.obsPrior.invW, Wk) )
      
      Hq_Lk   = -logNormConstWish( Wk, dFk)
      Hq_Lk  -= 0.5*(dFk - D - 1)*self.logLtilde[k] + 0.5*dFk*D
      
      Eq_mLk  = 0.5*self.logLtilde[k] + 0.5*D*np.log( self.qMixComp[k].beta/TWOPI )
      Eq_mLk -= Hq_Lk  #D/2 factor up top
      
      exk[k] = Ep_Xk
      Ep_X += 0.5*SS['N'][k]*Ep_Xk
      Ep_mL += 0.5*Ep_mLk
      Eq_mL += Eq_mLk

    print 'log|W|= ', ' '.join( ['%.4e'%x for x in self.detW] )        
    print 'log|L|= ', ' '.join( ['%.4e'%x for x in self.logLtilde] )  
    print 'xWx = ', ' '.join( ['%.4e'%x for x in xWx] )  
    print 'trSW =', ' '.join( ['%.4e'%x for x in SWtr] )
    print 'exk =', ' '.join( ['%.4e'%x for x in exk] )
    print 'Ep[ X ] = % .4e' % (Ep_X)  
    
    print 'Ep[ mL ] = % .4e' % (Ep_mL)  
    
    return Ep_X+ Ep_z+ Ep_w+ Ep_mL - Eq_z -Eq_w -Eq_mL
      
  def print_state( self, iterid, evBound, doFinal=False, status=''):
    doPrint = iterid % self.printEvery==0
    logmsg  = '  %5d/%d after %6.0f sec. | evidence % .6e' % (iterid, self.Niter, time.time()-self.start_time, evBound)
    if iterid ==0:
      print 'Initialized via %s.' % (self.initname)
      print logmsg
    elif (doFinal and not doPrint):
      print logmsg
    elif (not doFinal and doPrint):
      print logmsg
    if doFinal:
      print '... done. %s' % (status)

    
  def save_state( self, iterid, evBound ):
    if iterid==0: 
      mode = 'w' # create logfiles from scratch on initialization
    else:
      mode = 'a' # otherwise just append
      
    if iterid % (self.saveEvery)==0:
      filename, ext = os.path.splitext( self.savefilename )
      with open( filename+'.alpha', mode) as f:
        f.write( np2flatstr(self.alpha)+'\n')

      with open( filename+'.qObs', mode) as f:
        f.write( ' '.join( [str(q) for q in self.qMixComp]) +'\n' )
      
      with open( filename+'.iters', mode) as f:
        f.write( '%d\n' % (iterid) )
        
      with open( filename+'.evbound', mode) as f:
        f.write( '% .6e\n'% (evBound) )
