'''
BerkSegPatchData.py

  Streaming data generator that obtains square patches (with DC removed)
    from images in the Berkeley Segmentation dataset

  Author: Mike Hughes (mike@michaelchughes.com)
'''
import numpy as np
import glob
import os.path
import random
import time

DPATH = '/data/BSR/patches/train/'

DEXT  = '*.dat*'

def print_data_info():
  print 'Patches from the Berkeley Segmentation dataset (training)'

def get_data( seed=8675309, Nimg=2, **kwargs ):
  tstart = time.time()
  print '  Loading all patches from %d images...' % (Nimg)
  #X = np.loadtxt( '/data/BSR/patches/AllTrain.dat' )
  #print '    done after %.1f sec.' % (time.time() - tstart)
  #return X
  # old stuff below
  np.random.seed( seed )
  fList = glob.glob( os.path.join(DPATH, DEXT) )
  random.shuffle( fList )
  Xlist = list()
  for fname in fList[:Nimg:-1]:
    Xlist.append( np.loadtxt( fname ) )
  X = np.vstack( Xlist )
  print '    done after %.1f sec. %d patches loaded.' % (time.time() - tstart, X.shape[0])
  return X

def minibatch_generator(  batch_size=1000, nBatch=50, nRep=1, seed=8675309, **kwargs):
  for repID in range( nRep ):
    np.random.seed( seed )
    fList = glob.glob( os.path.join(DPATH, DEXT) )
    random.shuffle( fList )
    fList = fList[:2]
    Xcache = np.loadtxt( fList.pop() )
    for batchID in range( nBatch ):
      try:
        while Xcache.shape[0] < batch_size:
          Xcache = np.vstack( [Xcache, np.loadtxt( fList.pop()) ] )
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

