#! /home/mhughes/mypy/epd64/bin/python
#$ -S /home/mhughes/mypy/epd64/bin/python
# ------ set working directory
#$ -cwd 
# ------ attach job number
#$ -j n
# ------ send to particular queue
#$ -o ../logs/$JOB_ID.$TASK_ID.out
#$ -e ../logs/$JOB_ID.$TASK_ID.err

import os
import sys
import time

import numpy as np
import argparse
import sklearn.mixture
import importlib
from distutils.dir_util import mkpath

os.chdir('..')
sys.path[0] = os.getcwd()
print 'Python version %d.%d.%d' % sys.version_info[ :3]
print 'Numpy version %s' % (np.__version__)
print 'Cur Dir:', os.getcwd()
print 'Local search path:', sys.path[0]

class MyLogFile(object):
  def __init__(self, filepath):
    mkpath( os.path.split(filepath)[0] )
    self.file = open( filepath, 'w', 1 )

  def flush( self ):
    self.file.flush()

  def __getattr__(self, attr):
    return getattr( self.file, attr )

  def write( self, data):
    self.file.write( data )
    self.file.flush()
    os.fsync( self.file.fileno() )
 
  def fileno( self ):
    return self.file.fileno()

  def close( self ):
    self.file.close()

try:
  jobID = int(  os.getenv( 'JOB_ID' ) )
  taskID = int( os.getenv( 'SGE_TASK_ID' ) )
except TypeError:
  jobID = 1
  taskID = 1
  
p = argparse.ArgumentParser()
p.add_argument( 'data', type=str)
p.add_argument( 'infName', type=str)
p.add_argument( '--jobname', type=str, default='defaultSK' )
p.add_argument( '--nIter', type=int, default=10 )
p.add_argument( '--K', type=int, default=3 )

args, unknown = p.parse_known_args()

print 'JobID  %d' % (jobID )
print 'TaskID %d' % (taskID )

print 'Importing data module %s' % (args.data)
datagenmod = __import__( 'GMM.data.' + args.data, fromlist=['GMM','data'])

if 'print_data_info' in dir( datagenmod ):
  datagenmod.print_data_info()

outname = 'logs/%s/%s/%s-%04d.out' % (args.data[:7], args.infName, args.jobname, taskID)
errname = 'logs/%s/%s/%s-%04d.err' % (args.data[:7], args.infName, args.jobname, taskID)
print 'all future IO directed to ', outname
sys.stdout = MyLogFile( outname )
sys.stderr = MyLogFile( errname )



print 'Hello, this is job %s, id %d' % (args.jobname, jobID )

seed = hash( args.jobname + str(taskID) ) % np.iinfo(int).max
mygmm = sklearn.mixture.GMM(  n_components=args.K, random_state=seed, covariance_type='full', min_covar=1e-7, n_init=1, n_iter=args.nIter )

tstart = time.time()
Data    = datagenmod.get_data()    
mygmm.fit(  Data )
telapsed = time.time()-tstart
print ' %.2f sec total' % ( telapsed )
print ' %.2f sec /iter' % ( telapsed/args.nIter )

