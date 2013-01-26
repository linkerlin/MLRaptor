'''
EasyToyGMMDataGenerator

  Streaming data generator that draws samples from a simple GMM
     with 3 components, easily separable means and covariances

  Author: Mike Hughes (mike@michaelchughes.com)
'''
import numpy as np

K = 3
D = 5

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

def print_data_info():
  print 'Easy-to-learn toy data for K=3 GMM'
  print '  Mix weights:  '
  print '                ', np2flatstr( w )
  print '  Means:  '
  for k in range( K ):
    print '                ', np2flatstr( Mu[k] )

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
        Xk = Mu[k,:] + np.sqrt( Sigma[k,:] )*np.random.randn( Npercomp[k], D )
        X.append( Xk )
      X = np.vstack( X )
      yield X


def np2flatstr( X, fmt='% 7.2f' ):
  return ' '.join( [fmt % x for x in X.flatten() ] )  

