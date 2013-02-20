import argparse
import time
import numpy as np
from matplotlib import pylab

np.set_printoptions( linewidth=120, precision=6, suppress=True)

def load_mat_from_file( savefilename, varname, iterids=None ):
  fname = savefilename+'.'+varname
  mat   = np.loadtxt( fname )
  iters = np.loadtxt( savefilename+'.iters' )
  if iterids is None:
    return mat
  else:
    Dist = np.abs( iters[np.newaxis,:] - np.asarray(iterids)[:,np.newaxis] )
    mask = np.argmin( Dist , axis=1 )
    if mask.sum() == 0:
      raise Exception, 'Iteration not found'
    return mat[ mask ]
    
def extract_params_GaussWish( qobsFlat, D=5 ):
  kapFlat = qobsFlat[ :, 0]
  muFlat   = qobsFlat[ :, 1:D+1]
  dfFlat  = qobsFlat[ :, D+1].flatten()
  invWFlat = qobsFlat[ :, D+2:]

  sigmaFlat = invWFlat / ( dfFlat[:,np.newaxis] - D + 1 )
  
  return muFlat, sigmaFlat

def extract_w_from_stickparams( aFlat, K ):
  I = aFlat.shape[0]
  wFlat = np.empty( (I,K) )
  v = aFlat[:, K:] / ( aFlat[:, :K] + aFlat[:, K:] )

  print "V", v.shape
  for kk in xrange(K):
    wFlat[:,kk] = v[:,kk] * np.prod( 1 - v[:,:kk], axis=1 )
  
  print "W", wFlat.shape
  print wFlat[:3]
  print wFlat[:3].sum(axis=1)
  try:
    assert np.allclose( wFlat.sum(axis=1), 1.0, rtol=1e-4 )
  except AssertionError:
    print wFlat.sum(axis=1).max()
  return wFlat

def load_params( savefilename, iterids=None):
  iters = np.loadtxt( savefilename+'.iters' )

  if savefilename.count( 'EM/' ) > 0:
    wFlat = load_mat_from_file( savefilename, 'w', iterids=iterids)
    muFlat = load_mat_from_file( savefilename, 'mu', iterids=iterids)
    sigmaFlat = load_mat_from_file( savefilename, 'sigma', iterids=iterids)
  elif savefilename.count( 'VB/' ) > 0:
    aFlat = load_mat_from_file( savefilename, 'alpha', iterids=iterids)
    K = aFlat.shape[1]
    if savefilename.count( 'DPVB/')>0:
      K = K/2
      wFlat = extract_w_from_stickparams( aFlat, K)
    else:
      wFlat = aFlat / aFlat.sum(axis=1)[:,np.newaxis]
    for k in xrange(K):
      qobsFlat = load_mat_from_file( savefilename, 'qObsComp%03d' % (k), iterids=iterids)
      muF, sigmaF = extract_params_GaussWish( qobsFlat )
      if k == 0:
        muFlat = muF
        sigmaFlat = sigmaF
      else:
        muFlat = np.hstack( [muFlat, muF] )
        sigmaFlat = np.hstack( [sigmaFlat, sigmaF] )
    
  return wFlat, muFlat, sigmaFlat, iters

def plot_gaussian_ellipse( mu, Sigma, colorID=0, dimIDs=[0,1] ):
  assert len(dimIDs)==2 
  mu = np.asarray( mu )[dimIDs]
  Sigma = np.asarray( Sigma )
  if Sigma.ndim == 1:
    Sigma = np.diag( Sigma[dimIDs] )
  else:
    Sigma = Sigma[dimIDs, :]
    Sigma = Sigma[:, dimIDs]
  colorList = ['b','g','r','c','m','y','k','w']
  color = colorList[ colorID % len(colorList) ]
  lam, U = np.linalg.eig( Sigma )
  A = U * np.sqrt(lam)[np.newaxis,:]
  t = np.linspace( -np.pi, np.pi, 100)
  x = np.cos(t)
  y = np.sin(t)
  z = np.vstack( [x,y]  )  # 2 x len(t)
  for r in np.linspace( 0.1, 2, 4):
    zrot = r*np.dot( A, z )
    pylab.plot( mu[0]+zrot[0,:], mu[1]+zrot[1,:], color+'.-', markeredgecolor=color )
    pylab.hold(True)

def plot_param_trace( savefilename, dimIDs=[0,4], MIN_THR=0.03 ):
  wFlat, muFlat, sigmaFlat, iters = load_params( savefilename )
  nIter,K = wFlat.shape
  pylab.ion()
  for ii, iterid in enumerate(iters):
    D = muFlat[ii,:].size / K
    mu = muFlat[ii,:].reshape( (K,D) )
    try:
      Sigma = sigmaFlat[ii,:].reshape( (K,D) )
    except ValueError:
      Sigma = sigmaFlat[ii].reshape( (K,D,D) )
    pylab.hold(False)
    for k in range( K ):
      if wFlat[ii,k] > MIN_THR:
        plot_gaussian_ellipse( mu[k], Sigma[k], k, dimIDs=dimIDs )      
    pylab.xlabel( 'iter %4d'%(iterid) )
    pylab.axis( 'image' )
    pylab.draw()
    time.sleep(.1)


def plot_evidence_trace( savefilename ):
  evidence = load_mat_from_file( savefilename, 'evidence' )
  iters    = load_mat_from_file( savefilename, 'iters' )
  pylab.plot( iters, evidence, 'k+-' )
  pylab.show()

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument( 'savefilename', type=str )
    Parser.add_argument( '--evidence', action='store_true', default=False )
    Parser.add_argument( '--dimIDs', type=int, nargs='+', default=[0,1] )
    args = Parser.parse_args()

    if args.evidence:
      plot_evidence_trace( args.savefilename )
    else:
      plot_param_trace( args.savefilename, args.dimIDs )
    

if __name__ == '__main__':
  main()
