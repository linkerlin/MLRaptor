'''
EasyToyBernData

  Streaming data generator that draws samples from a simple 
     Bernoulli mixture model with 3 easy-to-separate components
     
  Author: Mike Hughes (mike@michaelchughes.com)
'''
import numpy as np

K = 3
D = 9

alpha = 0.5

#  w = [0.25, 0.25, 0.5]
w = 2**np.linspace( -K, -1, K)
w[0] = 1 - sum(w[1:])

# Phi = [.75 .75 .75 .2 .2 .2 .2]
FGprob = 0.75
BGprob = 0.2
Phi = BGprob*np.ones( (K,D) )
for k in xrange(K):
  favorIDs = np.arange(D/K) + D/K*k
  Phi[k, favorIDs] = FGprob

def sample_data_from_comp( k, Nk ):
  return np.float32( np.random.rand( Nk, D) < Phi[k,:] )
  
def print_data_info( modelName ):
  print 'Easy-to-learn toy data for K=3 Bernoulli Obs Model'
  print '  Mix weights:  '
  print '                ', np2flatstr( w )
  print '  Bern Probs:  '
  for k in range( K ):
    print '                ', np2flatstr( Phi[k] )

#######################################################################  Mixture data
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
#######################################################################  Admixture data
def get_data_by_groups( seed=8675309, nPerGroup=5000, nGroup=5, **kwargs):
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

    GroupIDs.append(  (gg*nPerGroup) + np.arange( 0, nPerGroup ) )
    X[ GroupIDs[gg] ] = Xgroup
    TrueW_perGroup.append( w )
  Data = dict()
  Data['X'] = X
  Data['GroupIDs'] = GroupIDs
  Data['nGroup']   = nGroup
  Data['TrueW_perGroup']   = np.vstack( TrueW_perGroup)
  return Data

def group_minibatch_generator( batch_size=1000,nBatch=50,nRep=1,seed=8675309,**kwargs):
  for repID in range( nRep ):
    np.random.seed( seed )
    for batchID in range( nBatch ):
      Data = get_data_by_groups( seed=seed+batchID, nPerGroup=batch_size, nGroup=5)
      yield Data

def np2flatstr( X, fmt='% 7.2f' ):
  return ' '.join( [fmt % x for x in X.flatten() ] )  

