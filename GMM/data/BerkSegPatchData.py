'''
BerkSegPatchData.py

  Streaming data generator that obtains square patches (with DC removed)
    from images in the Berkeley Segmentation dataset

  Author: Mike Hughes (mike@michaelchughes.com)
'''
import numpy as np
import glob
import os.path
import sys
import random
import time

# Read in the data directory from file
with open( os.path.join('GMM','data','BerkSegPatchData.path'),'r') as f:
  DPATH = f.readline().strip()
if not os.path.exists( DPATH ):
  raise Exception, 'Specified path %s is not valid!\n' \
                   'Please edit the file BerkSegPatchData.path' % (DPATH)

DEXT  = '*.dat*'


def print_data_info():
  print 'Patches from the Berkeley Segmentation dataset (training)'
  fList = glob.glob( os.path.join(DPATH, DEXT) )
  X = np.loadtxt( fList.pop() )
  D = np.sqrt( X.shape[1] )
  print ' %d images.  ~ %d patches / image.  patch size: %d x %d pixels' \
              % (len(fList), X.shape[0], D,D )

def get_data( seed=8675309, Nimg=2, **kwargs ):
  tstart = time.time()
  print '  Loading all patches from %d images...' % (Nimg)

  random.seed( seed )
  fList = glob.glob( os.path.join(DPATH, DEXT) )
  random.shuffle( fList )
  Xlist = list()
  for fname in fList[:Nimg]:
    Xlist.append( np.loadtxt( fname ) )
  X = np.vstack( Xlist )
  print '    done after %.1f sec. %d patches loaded.' % (time.time() - tstart, X.shape[0])
  return X

def minibatch_generator(  batch_size=1000, nBatch=50, nRep=1, seed=8675309, **kwargs):
  for repID in range( nRep ):
    random.seed( seed )
    fList = glob.glob( os.path.join(DPATH, DEXT) )
    random.shuffle( fList )
    Xcache = np.loadtxt( fList.pop(0) )
    for batchID in range( nBatch ):
      try:
        while Xcache.shape[0] < batch_size:
          Xcache = np.vstack( [Xcache, np.loadtxt( fList.pop(0)) ] )
      except IndexError:
        #print 'returning entire cache:', Xcache.shape[0]
        yield Xcache
        break 
      Xcur = Xcache[:batch_size]
      Xcache = Xcache[batch_size:]
      #print 'remaining cache size:', Xcache.shape[0]
      yield Xcur

def np2flatstr( X, fmt='% 7.2f' ):
  return ' '.join( [fmt % x for x in X.flatten() ] )  

