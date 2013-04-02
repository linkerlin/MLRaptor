'''
TrickyToyGaussData

  Streaming data generator that draws samples from a tough-to-learn GMM
     with 16 components

  Author: Mike Hughes (mike@michaelchughes.com)
'''
import scipy.linalg
import numpy as np

import TrickyToyGaussData as TT

K = 20
D = 2
alpha = 0.5

w = np.tile( [4.,3.,4.,3.,1.], K/TT.K )
w = w/w.sum()

Mu = np.zeros( (K,D) )
Mu[ :5,:] = [1.0,1.0]
Mu[5:10,:] = [-1.0, 1.0]
Mu[10:15]  = [-1.0, -1.0]
Mu[15:,:]   = [1.0, -1.0]

Sigma = np.zeros( (K,D,D) )
cholSigma = np.zeros( Sigma.shape )
for k in xrange( K ):
  Sigma[k] = TT.Sigma[ k % 5]
  cholSigma[k] = scipy.linalg.cholesky( Sigma[k] )

######################################################################  Module Util Fcns
def sample_data_from_comp( k, Nk ):
  return Mu[k,:] + np.dot( cholSigma[k].T, np.random.randn( D, Nk) ).T

def get_short_name( ):
  return 'TrickyBig'

def print_data_info( mName ):
  print 'Tricky Toy Data. K=%d. D=%d.' % (K,D)

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
