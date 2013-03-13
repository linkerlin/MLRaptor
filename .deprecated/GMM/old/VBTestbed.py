import scipy
import numpy as np
import EasyToyGMMDataGenerator as Toy
import GMM as GMM
import GMMPrior as OP
import VBLearnerGMM as VB

doGeo = True

def np2flatstr( X, fmt=None ):
  if fmt is None:
    if np.max(np.abs(X)) > 0.1 and np.max( np.abs(X) ) <= 10000:
      fmt = '% 8.3f'
    else:
      fmt = '% 7.3e'
  s = ''
  if X.ndim==1:
    X = X.reshape( (1,X.size) )
  for rowID in range( X.shape[0] ):
    s += ' '.join( [fmt % x for x in X[rowID].flatten() ] )  
    s += '\n'
  return s
  
if doGeo:
  X = np.loadtxt( 'GeoData.txt')
else:
  DG = Toy.minibatch_generator( batch_size=10000, nBatch=1 )
  X = DG.next()

N = X.shape[0]
K = 4
alph = 1.0
D = X.shape[1]
dF = D+2
invW = np.eye(D)
precMu = 1.0 

gmm = GMM.GMM( K, covariance_type='full' )
gmm.D = D
op  = OP.GMMPrior( dF, invW, precMu )

vb = VB.VBLearnerGMM( gmm, alph, op, printEvery=1, Niter=10 )

Dist = scipy.spatial.distance.cdist( X, X[:4] )
labelIDs = np.argmin( Dist, axis=1 )
R = np.float64( np.tile( np.arange(K), (N,1) ) == labelIDs[:,np.newaxis] )
print R[:5,:]

SS = vb.calc_suff_stats( X, R )
vb.M_step( SS )

print 'mean Params'
for k in range(K):
  print np2flatstr( vb.qMixComp[k].m )
  
print 'invW Params'
for k in range(K):
  print np2flatstr( vb.qMixComp[k].invW )
  
print 'dF Params'
for k in range(K):
  print vb.qMixComp[k].dF
  
  
print 'beta Params'
for k in range(K):
  print vb.qMixComp[k].beta
  
vb.calc_ELBO( R, SS )
