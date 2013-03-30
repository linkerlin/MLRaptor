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
import platform

# Read in the data directory from file
if platform.system() == 'Darwin':
  DATADIRNAME  = os.path.join( '/Users/mhughes/git/MLRaptor/data' )
else:
  DATADIRNAME  = os.path.join( '/home/mhughes/git/MLRaptor/data' )
PATHFILENAME = os.path.join(DATADIRNAME,'BerkSegPatchData.path')
with open( PATHFILENAME,'r') as f:
  DPATH = f.readline().strip()
if not os.path.exists( DPATH ):
  raise Exception, 'Specified path %s is not valid!\n' \
                   'Please edit the file BerkSegPatchData.path' % (DPATH)

DEXT  = '*.dat*'

NimgDEF = 50

def print_data_info( modelName, **kwargs):
  print 'Data: Patches from the Berkeley Segmentation dataset (training)'
  fList = glob.glob( os.path.join(DPATH, DEXT) )
  X = np.loadtxt( fList.pop() )
  D = np.sqrt( X.shape[1] )
  print ' %d images.  ~ %d patches / image.  patch size: %d x %d pixels' \
              % (len(fList), X.shape[0], D,D )

def get_data( seed=8675309, Nimg=NimgDEF, **kwargs ):
  tstart = time.time()
  print '  Loading all patches from %d images...' % (Nimg)

  random.seed( seed )
  fList = glob.glob( os.path.join(DPATH, DEXT) )
  random.shuffle( fList )
  fList = fList[:Nimg]
  Xlist = list()
  for fname in fList:
    Xlist.append( np.loadtxt( fname ) )
  X = np.vstack( Xlist )
  print '    done after %.1f sec. %d patches loaded.' % (time.time() - tstart, X.shape[0])
  return {'X':X}

def minibatch_generator(  batch_size=1000, Nimg=NimgDEF, nBatch=50, nRep=1, seed=8675309, **kwargs):
  for repID in range( nRep ):
    random.seed( seed )
    fList = glob.glob( os.path.join(DPATH, DEXT) )
    random.shuffle( fList )
    fList = fList[:Nimg]
    Xcache = np.loadtxt( fList.pop(0) )
    for batchID in range( nBatch ):
      try:
        while Xcache.shape[0] < batch_size:
          Xcache = np.vstack( [Xcache, np.loadtxt( fList.pop(0)) ] )
      except IndexError:
        #print 'returning entire cache:', Xcache.shape[0]
        yield {'X':Xcache}
        break 
      Xcur = Xcache[:batch_size]
      Xcache = Xcache[batch_size:]
      #print 'remaining cache size:', Xcache.shape[0]
      yield {'X':Xcur}

def load_image_as_group_data( fname, startID=0 ):
  X = np.loadtxt( fname )
  GroupID = (startID, startID+X.shape[0] )
  curID = startID + X.shape[0]
  return X, GroupID, curID

################################################################################## ADMIXTURE LOAD
def get_data_by_groups( seed=8675309, Nimg=NimgDEF, **kwargs ):
  tstart = time.time()
  print '  Loading all patches from %d images (grouped by image)...' % (Nimg)
  random.seed( seed )
  fList = glob.glob( os.path.join(DPATH, DEXT) )
  random.shuffle( fList )
  fList = fList[:Nimg]
  Xlist = list()
  GroupIDs = list()
  nTotal = 0
  for fname in fList:
    Xcur, curID, nTotal = load_image_as_group_data( fname, nTotal )
    Xlist.append( Xcur )
    GroupIDs.append( curID )
  X = np.vstack( Xlist )
  print '    done after %.1f sec. %d patches loaded.' % (time.time() - tstart, X.shape[0])
  return {'X':X, 'GroupIDs':GroupIDs, 'nGroup':len(GroupIDs)}

def group_minibatch_generator(  batch_size=9401, nBatch=50, nRep=1, seed=8675309, **kwargs):
  for repID in range( nRep ):
    random.seed( seed )
    fList = glob.glob( os.path.join(DPATH, DEXT) )
    random.shuffle( fList )
    fList = fList[:nBatch]
    Xlist = list()
    GroupIDs = list()
    nTotal = 0
    for batchID in range( nBatch ):
      Xcur, curID, nTotal = load_image_as_group_data( fList[batchID], nTotal )
      Xlist.append( Xcur )
      GroupIDs.append( curID )    
      if nTotal >= batch_size:  
        yield dict( X=np.vstack(Xlist), GroupIDs=GroupIDs, nGroup=len(GroupIDs) )
        Xlist=list()
        GroupIDs=list()
        nTotal=0
    if nTotal > 0:
      yield dict( X=np.vstack(Xlist), GroupIDs=GroupIDs, nGroup=len(GroupIDs) )
   

def np2flatstr( X, fmt='% 7.2f' ):
  return ' '.join( [fmt % x for x in X.flatten() ] )  

