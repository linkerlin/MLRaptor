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
  
def print_data_info():
  print 'Easy-to-learn toy data for K=3 Bernoulli Obs Model'
  print '  Mix weights:  '
  print '                ', np2flatstr( w )
  print '  Bern Probs:  '
  for k in range( K ):
    print '                ', np2flatstr( Phi[k] )


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

def np2flatstr( X, fmt='% 7.2f' ):
  return ' '.join( [fmt % x for x in X.flatten() ] )  

