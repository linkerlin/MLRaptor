import numpy as np
import scipy.optimize
from scipy.special import digamma, gammaln
EPS = 10*np.finfo(float).eps

def v2beta( v ):
  v = np.hstack( [v, 1] )
  c1mv = np.cumprod( 1 - v )
  c1mv = np.hstack( [1, c1mv] )
  beta = v * c1mv[:-1]
  return beta

def dbetadv( v, beta):
  K = v.size
  dbdv = np.zeros( (K, K+1) )
  for k in xrange( K ):
    dbdv[k, k] = beta[k]/v[k]
    dbdv[k, k+1:] = -1.0*beta[k+1:]/(1-v[k])
  return dbdv
  
def neglogp( v, G, logpiMat, alpha0, gamma ):
  ''' Compute negative log posterior prob of v
        up to an additive constant
  '''
  try:
    assert np.all( v >= 0)
    assert np.all( v <= 1 )
  except AssertionError:
    return np.inf

  beta = v2beta(v)
  logp = -G * np.sum( gammaln( gamma*beta+EPS ) )
  logp += gamma*np.sum( beta * logpiMat )
  logp += (alpha0-1)*np.sum( np.log( 1- v ) )
  return -1.0*logp

def gradneglogp( v, G, logpiMat, alpha0, gamma ):
  ''' Compute gradient log posterior prob of v
        up to an additive constant
  '''
  try:
    assert np.all( v >= 0)
    assert np.all( v <= 1 )
  except AssertionError:
    return np.nan*np.ones( v.size )

  beta = v2beta(v)
  beta = np.maximum(beta,EPS)
  beta = np.minimum(beta,1-EPS)
  dBdv = dbetadv( v, beta )

  ZDir_grad = gamma* np.dot( dBdv, digamma(gamma*beta) )
  gradvec = -1.0*(alpha0-1.0)/(1.0-v)
  gradvec -= G*ZDir_grad
  gradvec += gamma* np.dot( dBdv, logpiMat)

  return -1.0*gradvec

def get_best_stick_prop_point_est( K, G, Elogw, alpha0, gamma, vinitIN=None, LB=1e-7, Ntrial=5):
  objfunc = lambda v: neglogp( v, G, Elogw, alpha0, gamma)
  objgrad = lambda v: gradneglogp( v, G, Elogw, alpha0, gamma)
 
  Bounds = [ (LB, 1-LB) for k in xrange(K) ]
  for trial in xrange(Ntrial):
    if vinitIN is None:
      vinit = np.random.rand( K )
    else:
      vinit = 0.0001*np.random.randn(K) + vinitIN
      vinit = np.maximum( vinit, LB)
      vinit = np.minimum( vinit, 1-LB)
    finit = objfunc(vinit)
    v,f,d = scipy.optimize.fmin_l_bfgs_b( objfunc, x0=vinit, fprime=objgrad, bounds=Bounds)
    if check_bounds( v, f, finit, LB):
      return v
  return vinit  
  
def check_bounds(x, f, finit, LB):
  isGood = np.all( x >= LB )
  isGood = isGood and np.all( x <= 1-LB )
  return isGood
