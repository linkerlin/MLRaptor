'''
TrickyToyGaussData

  Streaming data generator that draws samples from a tough-to-learn GMM
     with 16 components

  Author: Mike Hughes (mike@michaelchughes.com)
'''
import scipy.linalg
import numpy as np

K = 5
D = 2
alpha = 0.5

######################################################################  Generate Toy Params
w = np.asarray( [5., 4., 3., 2., 1.] )
w = w/w.sum()

Mu = np.zeros( (K,D) )

theta = np.pi/4
RotMat = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
RotMat = np.asarray( RotMat)
varMajor = 1.0/16.0
SigmaBase = [[ varMajor, 0], [0, varMajor/100.0]]
SigmaBase = np.asarray(SigmaBase)

Lam,V = np.linalg.eig( SigmaBase )
Lam = np.diag(Lam)
Sigma = np.zeros( (5,D,D) )
for k in xrange(4):
  Sigma[k] = np.dot( V, np.dot( Lam, V.T) )
  V = np.dot( V, RotMat )
Sigma[4] = 1.0/5.0*np.eye(D)

cholSigma = np.zeros( Sigma.shape )
for k in xrange( K ):
  cholSigma[k] = scipy.linalg.cholesky( Sigma[k] )

######################################################################  Module Util Fcns
def sample_data_from_comp( k, Nk ):
  return Mu[k,:] + np.dot( cholSigma[k].T, np.random.randn( D, Nk) ).T

def get_short_name( ):
  return 'TrickyToy'

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
