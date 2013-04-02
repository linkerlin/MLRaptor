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

import FloReader as FR

# Read in the data directory from file
DPATH = '/data/liv/visiondatasets/sintel/patches/'
DEXT  = '*/*.dat*'

def get_fpath_list():
  return sorted( glob.glob( DPATH+DEXT) )

def load_frame_flow_data_data( fname, startID=0 ):
  X = np.loadtxt( fname )
  GroupID = (startID, startID+X.shape[0] )
  curID = startID + X.shape[0]
  return X, GroupID, curID

def get_short_name():
  return 'Sintel'

def print_data_info( modelName, **kwargs):
  print 'Data: Patches from the Sintel Optical Flow dataset (training)'
  fList = get_fpath_list()
  X = np.loadtxt( fList[0] )
  D = np.sqrt( X.shape[1]/2 )
  print ' %d frames.  ~ %d patches / frame.  Patch size: %d x %d pixels' \
              % (len(fList), X.shape[0], D,D )

def get_data( seed=8675309, **kwargs ):
  tstart = time.time()
  fList = get_fpath_list()
  print '  Loading all patches from %d frames...' % ( len(fList) )
  random.seed( seed )
  random.shuffle( fList )

  Xlist = list()
  for fname in fList:
    Xlist.append( np.loadtxt( fname ) )
  X = np.vstack( Xlist )
  print '    done after %.1f sec. %d patches loaded.' % (time.time() - tstart, X.shape[0])
  return {'X':X}

def minibatch_generator(  batch_size=1000, nBatch=50, nRep=1, seed=8675309, **kwargs):
  for repID in range( nRep ):
    fList = get_fpath_list()
    random.seed( seed )
    random.shuffle( fList )
    Xcache = np.loadtxt( fList.pop(0) )
    for batchID in range( nBatch ):
      try:
        while Xcache.shape[0] < batch_size:
          Xcache = np.vstack( [Xcache, np.loadtxt( fList.pop(0)) ] )
      except IndexError:
        yield {'X':Xcache}
        break 
      Xcur = Xcache[:batch_size]
      Xcache = Xcache[batch_size:]
      yield {'X':Xcur}

################################################################################## ADMIXTURE LOAD
## TO DO
   

def np2flatstr( X, fmt='% 7.2f' ):
  return ' '.join( [fmt % x for x in X.flatten() ] )  

