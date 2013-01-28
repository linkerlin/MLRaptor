import argparse
import time
import numpy as np
from matplotlib import pylab


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
    
def load_params( savefilename, iterids=None):
  iters = np.loadtxt( savefilename+'.iters' )
  wFlat = load_mat_from_file( savefilename, 'w', iterids=iterids)

  muFlat = load_mat_from_file( savefilename, 'mu', iterids=iterids)
  sigmaFlat = load_mat_from_file( savefilename, 'sigma', iterids=iterids)

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
  t = np.linspace( -np.pi, np.pi, 1000)
  x = np.cos(t)
  y = np.sin(t)
  z = np.vstack( [x,y]  )  # 2 x len(t)
  for r in np.linspace( 0.1, 2, 4):
    zrot = r*np.dot( A, z )
    pylab.plot( mu[0]+zrot[0,:], mu[1]+zrot[1,:], color+'.-', )
    pylab.hold(True)

def plot_param_trace( savefilename, dimIDs=[0,4] ):
  wFlat, muFlat, sigmaFlat, iters = load_params( savefilename )
  nIter,K = wFlat.shape
  pylab.ion()
  for ii, iterid in enumerate(iters):
    D = muFlat[ii,:].size / K
    mu = muFlat[ii,:].reshape( (K,D) )
    Sigma = sigmaFlat[ii,:].reshape( (K,D) )
    pylab.hold(False)
    for k in range( K ):
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
