'''
EasyToyGMMData

  Streaming data generator that draws samples from a simple GMM
     with 3 components, easily separable means and covariances

  Author: Mike Hughes (mike@michaelchughes.com)
'''
import numpy as np

K = 3
D = 5

alpha = 0.5

#  w = [0.25, 0.25, 0.5]
w = 2**np.linspace( -K, -1, K)
w[0] = 1 - sum(w[1:])

# Mu = [ -10 0
#          0 0
#        +10 0]
Mu = np.zeros( (K,D) )
Mu[:,0] = np.linspace(-10-10*(K-3), 10+10*(K-3), K )

# Sigma = [0 0.5
#          0 1.0
#          0 2.0]
Sigma   = np.ones( (K,D) )
Sigma[:,-1] = 2**np.linspace( -K+2, 1, K)

PiInit = 1.0/K * np.ones(K)
PiMat  = alpha + np.eye( K )
PiMat[0,1] = alpha/5
PiMat[1,2] = alpha/10
PiMat[2,0] = alpha/3
PiMat /= PiMat.sum(axis=1)[:,np.newaxis]


def sample_data_from_comp( k, Nk ):
  return Mu[k,:] + np.sqrt( Sigma[k,:] )* np.random.randn( Nk, D)

def print_data_info( modelName):
  print 'Easy-to-learn toy data for K=3 GMM'
  if modelName.count('Admix') > 0:
    print '  Dir. prior on group mix weights:', alpha
  elif modelName.count('HMM') > 0:
    print '  Trans. matrix:  '
    for k in xrange(K):
      print '                ', np2flatstr( PiMat[k] )
  else:
    print '  Mix weights:  '
    print '                ', np2flatstr( w )
  print '  Means:  '
  for k in range( K ):
    print '                ', np2flatstr( Mu[k] )

######################################################################  MixModel Data
def get_data( seed=8675309, **kwargs ):
  DG = minibatch_generator( batch_size=25000, nBatch=1, seed=seed)
  X = DG.next()
  return X

def minibatch_generator(  batch_size=1000, nBatch=50, nRep=1, seed=8675309, **kwargs):
  for repID in range( nRep ):
    np.random.seed( seed )
    for batchID in range( nBatch ):
      Npercomp = np.random.mtrand.multinomial( batch_size, w )
      X = list()
      for k in range(K):
        X.append( sample_data_from_comp( k, Npercomp[k]) )
      X = np.vstack( X )
      yield {'X':X}

######################################################################  HMM Data
def gen_discrete_Markov_seq( PiInit, PiMat, T):
  z = np.zeros( T )
  z[0] = discrete_single_draw( PiInit )
  for tt in xrange(1,T):
    z[tt] = discrete_single_draw( PiMat[z[tt-1]] )
  return z

def get_sequence_data( seed=8675309, nSeq=50, T=1000, **kwargs ):
  np.random.seed( seed )
  X = np.zeros( (nSeq*T, D) )
  Tstart = list()
  Tstop = list()
  tall = 0
  for ii in xrange( nSeq ):
    z = gen_discrete_Markov_seq( PiInit, PiMat, T)
    for tt in xrange( T ):
      X[ tall+tt ] = sample_data_from_comp( z[tt], 1 )
    Tstart.append( tall)
    Tstop.append( tall+T )
    tall += T
  return dict( X=X, nSeq=nSeq, Tstart=np.asarray(Tstart), Tstop=np.asarray(Tstop), TruePiMat=PiMat )

######################################################################  Admix data

def group_minibatch_generator( batch_size=1000,nBatch=50,nRep=1,seed=8675309,**kwargs):
  for repID in range( nRep ):
    np.random.seed( seed )
    for batchID in range( nBatch ):
      Data = get_data_by_groups( seed=seed+batchID, nPerGroup=batch_size, nGroup=5)
      yield Data  

def get_data_by_groups( seed=8675309, nPerGroup=5000, nGroup=5, **kwargs ):
  np.random.seed( seed )
  X = np.empty( (nGroup*nPerGroup, D) )
  GroupIDs = list()
  TrueW_perGroup = list()
  for gg in xrange( nGroup ):
    w = np.random.mtrand.dirichlet( alpha*np.ones(K)  )
    nPerComp = np.random.mtrand.multinomial( nPerGroup, w )
    Xlist = list()
    for k in xrange(K):
      Xlist.append(  sample_data_from_comp( k, nPerComp[k])  )
    Xgroup = np.vstack( Xlist )

    GroupIDs.append(  (gg*nPerGroup, gg*nPerGroup+nPerGroup)  )
    X[ GroupIDs[gg][0]:GroupIDs[gg][1] ] = Xgroup
    TrueW_perGroup.append( w )
  Data = dict()
  Data['X'] = X
  Data['GroupIDs'] = GroupIDs
  Data['nGroup']   = nGroup
  Data['TrueW_perGroup']   = np.vstack( TrueW_perGroup)
  return Data

######################################################################  Utils
def np2flatstr( X, fmt='% 7.2f' ):
  return ' '.join( [fmt % x for x in X.flatten() ] )  

def discrete_single_draw( ps):
  ''' Given vector of K weights "ps",
         draw a single integer assignment in {1,2, ...K}
      such that Prob( choice=k) = ps[k]
  '''
  totals = np.cumsum(ps)
  norm = totals[-1]
  throw = np.random.rand()*norm
  return np.searchsorted(totals, throw)

