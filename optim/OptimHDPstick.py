import time
import numpy as np
import scipy.optimize
from scipy.special import digamma, gammaln

EPS = 10*np.finfo(float).eps
np.set_printoptions( linewidth=120, precision=3, suppress=True )

def generate_v( K, alpha0):
  v = np.random.mtrand.beta( 1, alpha0, (K) ) # draw K stick fractions
  return v

def generate_group_mix_weights( G, beta, gamma ):
  pi = np.empty( (G, beta.size) )
  for gg in xrange( G ):
    pi[gg] = np.random.mtrand.dirichlet( gamma*beta )
  return pi

def v2beta( v ):
  v = np.hstack( [v, 1] )
  c1mv = np.cumprod( 1 - v )
  c1mv = np.hstack( [1, c1mv] )
  beta = v * c1mv[:-1]
  return beta

def beta2v( beta):
  cumB = np.hstack( [0,np.cumsum(beta[:-1])] )
  v = beta/(1-cumB)
  return v[:-1]

###############################################  Computing objective and derivatives
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

############################################################ Check neglogp function is correct

def check_random_draws( vOPT, *args):
  # Generate random betas and make sure they are worse
  didPass = True
  print 'Random check'
  nlogpOPT = neglogp(vOPT, *args)
  print vOPT, '  %.4e | Truth' % ( nlogpOPT )  
  for n in range(10):
    randvec = np.random.mtrand.dirichlet( np.ones(vOPT.size) )
    nlogp   = neglogp( randvec, *args )   
    print randvec, '  %.4e' % ( nlogp )
    if nlogp < nlogpOPT:
      print 'OH NO!**************'
      didPass = False
  return didPass

def check_perturbed_draws( vOPT, *args):
  # Generate random betas by small perturbations of optimal beta,
  #  and make sure these are (mostly) worse [have lower logp than optimal]
  didPass = True
  print 'Perturb check'
  for n in range(10):
    v = vOPT + 0.005*np.random.randn( vOPT.size )
    v = np.minimum( v, 1-1e-9)
    v = np.maximum( v, 1e-9)
    nlogp   = neglogp( v, *args )   
    print v, '  %.4e' % ( nlogp )
    if nlogp < nlogpOPT:
      print 'OH NO!**************'
      didPass = False
  return didPass

def np2flatstr( X ):
  bigstr = ' '.join( ['%.4f'% (x) for x in X] )
  if X.size < 10:
    return bigstr
  else:
    biglist = bigstr.split(' ')
    mystr = '\n'
    for m in np.arange( np.minimum(3,np.ceil(X.size / 10.0)) ):
      myinds = m*10 + np.arange( 0, 10 )
      mystr += ' '.join( [biglist[ int(ii) ] for ii in myinds if ii < X.size] )
      mystr += '\n'
      if m==0: 
        remstr = ' '*len( mystr )
    return mystr+remstr

def projBounds( x, LB, UB):
  x = np.minimum(x,UB-1e-8)
  x = np.maximum(x,LB+1e-8)
  return x

G = 100000
K = 250
alpha0 = K/2.0
gamma  = 50 # high value means pi has low variance.  good for recovering beta


def gen_problem( G, K, alpha0, gamma, seed):
  np.random.seed( seed )
  v       = generate_v( K, alpha0 )
  betavec = v2beta( v)
  piMat   = generate_group_mix_weights( G, betavec, gamma)
  logpiMat = np.sum( np.log(piMat+EPS), axis=0)
  args = (G, logpiMat, alpha0, gamma)
  return {'objfunc':neglogp, 'objgrad':gradneglogp, 'vTrue':v, 'betaTrue':betavec, 'args':args, 'piMat':piMat, 'logpi': logpiMat}

WARN_MSG = {0:'!!!', 1:'   '}


def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--G', type=int, default=10000)
  parser.add_argument('--K', type=int, default=10)
  parser.add_argument('--gamma', type=float, default=25)
  parser.add_argument('--alpha0', type=float, default=None)

  parser.add_argument('--Ntrial', type=int, default=5)
  parser.add_argument('--initname', type=str, default='rand')

  args=  parser.parse_args()

  if args.alpha0 is None:
    args.alpha0= args.K/2

  Problem = gen_problem( args.G, args.K, args.alpha0, args.gamma)
  optEngine = FractionVecOptimizer( Problem, LB=1e-6 ) 
  optEngine.run_many_trials( args.Ntrial, args.initname)

if __name__ == '__main__':
  main()

'''
print '---------------'
print np2flatstr(v)
print np2flatstr( projBounds(vOPT,LB,UB) )

print v.size, vOPT.size
assert np.allclose( v, vOPT)
print '---------------'

fOPT = objfunc(v)
print '%s | %.4e | Truth' % ( np2flatstr(betaOPT), fOPT )
nfail = 0
for trial in xrange( Ntrial ):
  #xinit = np.random.rand( K )
  #xinit  = 0.05*np.random.randn(K) + beta2v( np.mean( piMat, axis=0) )
  xinit = 0.01*np.random.randn(K) + vOPT
  xinit = projBounds( xinit, LB, UB )
  x,f,d = scipy.optimize.fmin_l_bfgs_b( objfunc, x0=xinit, fprime=objgrad, bounds=Bounds, m=25, factr=1e-25, pgtol=1e-8)
  if d['warnflag'] == 2:
    print '\t\t\t\t\t\t\t %s' % (d['task'])
  
  finit = objfunc(xinit)
  if f >= finit or np.allclose(f, finit):
    print 'WARNING: INIT = FINAL'

  rep = 0
  while np.sum( x == LB ) > 0 and rep < 100:
    xinit = x
    #xinit[ x==LB ] = 0.1 
    xinit = projBounds( xinit, LB, UB)
    x,f,d = scipy.optimize.fmin_l_bfgs_b( objfunc, x0=xinit, fprime=objgrad, bounds=Bounds, m=25, factr=1e-10, pgtol=1e-8)
    rep += 1
  status = 'rand + %d reps' % (rep)
    
  try:
    assert np.all( x >= LB )
    assert np.all( x <= UB )
  except AssertionError:
    x = projBounds(xinit,LB,UB)

  beta = v2beta( x )
  
  if f < fOPT or np.allclose( f, fOPT):
    print '%s | %.4e    | %s | %s' % ( np2flatstr(beta), f, status, d['task'] )
  else:
    print '%s | %.4e*** | %s | %s' % ( np2flatstr(beta), f, status, d['task'] )
    nfail += 1

print 'Total: %d/%d successful.' % (Ntrial-nfail, Ntrial)
'''


  
