#! /usr/bin/python

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import argparse

def invcdfmvnormal(p):
  return np.sqrt( -2*np.log(1-p) )

def plotEllipseFromMuSigma(mu, Sigma ):
  # need to flip from lat,lon to lon,lat
  mu = mu[::-1]
  tmp = Sigma[0,0]
  Sigma[0,0] = Sigma[1,1]
  Sigma[1,1] = tmp
  lam, V = np.linalg.eig( Sigma )
  V = V*np.sqrt(lam)
  ts = np.arange( -np.pi, np.pi, 0.01 )
  xs = list()
  ys = list()
  # plot elliptical contour curves such that
  #  1st curve (innermost) contains 25% of probability mass
  #  2nd curve contains 50%,
  #  etc.
  prctiles = np.asarray( [0.25, 0.5, 0.75, 0.99] )
  Rs = invcdfmvnormal( prctiles )
  for R in Rs:
    x = R* np.sin(ts)
    y = R* np.cos(ts)
    zs = np.dot(V, np.vstack( [x, y] ) )
    xs.extend( mu[0] + zs[0,:] )
    ys.extend( mu[1] + zs[1,:] )
  return (xs,ys)
  
Props = dict()
Props['llcrnrlat'] =20
Props['llcrnrlon'] =-125
Props['urcrnrlat'] = 49
Props['urcrnrlon'] = -60

def create_USA_map(): 
  lon_center = -95
  lat_center = 35
  EARTH_RADIUS_METERS = 6371200
  # create polar stereographic Basemap instance.
  m = Basemap(projection='stere',lon_0=lon_center,lat_0=lat_center, rsphere=EARTH_RADIUS_METERS,resolution='l',area_thresh=5000, **Props)
  m.drawcoastlines()
  m.drawstates()
  m.drawcountries()
  m.drawmapboundary(fill_color='#99ffff')
  m.fillcontinents(color='#cc9966',lake_color='#99ffff')
  m.drawparallels(np.arange(Props['llcrnrlat'],Props['urcrnrlat'],5),labels=[1,1,0,0])
  m.drawmeridians(np.arange( Props['llcrnrlon'],Props['urcrnrlon'],5),labels=[0,0,0,1])
  return m

def plotSavedGMMParams( mapfig, matfilename ):
  Mdict = scipy.io.loadmat( matfilename )
  Mu = Mdict['mu']
  Sigma = Mdict['Sigma']
  K = Mu.shape[0]
  for k in range( K ):
    mukk = Mu[ k,: ]
    Sigkk = Sigma[:,:, k ]
    if Sigkk.shape != (2,2):
      Sigkk = np.diag( Sigkk )
    xs, ys = plotEllipseFromMuSigma( mukk, Sigkk )
    xs,ys = mapfig( xs,ys) # translate from lon,lat to map coords
    mapfig.plot(xs,ys, 'b-' ) 

def plotGeoTagLocations( mapfig, matfilename ):
  X = scipy.io.loadmat( matfilename)['X']
  assert X.shape[0] > 0
  X += 1e-5*np.random.randn( X.shape[0], X.shape[1] )
  lons = X[:,1]
  lats = X[:,0]
  x,y = mapfig( lons, lats )
  mapfig.plot( x, y, 'r.' )
            
def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument( 'modelfilename', type=str )
    Parser.add_argument( '--datafilename', type=str, default='FlickrUSAGeoData.mat' )
    args = Parser.parse_args()
    m = create_USA_map()
    plotGeoTagLocations( m, args.datafilename )
    plotSavedGMMParams( m, args.modelfilename )
    plt.show()
    
if __name__ == '__main__':
    main()
