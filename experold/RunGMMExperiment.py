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
from distutils.dir_util import mkpath
import sys
import argparse
import subprocess
import platform
import numpy as np

os.chdir('..')
sys.path[0] = os.getcwd()
print 'Python version %d.%d.%d' % sys.version_info[ :3]
print 'Numpy version %s' % (np.__version__)
print 'Cur Dir:', os.getcwd()
print 'Local search path:', sys.path[0]

if platform.machine().count( '64' ) > 0:
  pyCMD = '/home/mhughes/mypy/epd64/bin/python'
else:
  pyCMD = '/home/mhughes/mypy/epd32/bin/python'

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
p.add_argument( '--jobname', type=str )

args, unknown = p.parse_known_args()

print 'JobID  %d' % (jobID )
print 'TaskID %d' % (taskID )

if not sys.stdout.isatty():
  outname = 'logs/%s/%s/%s-%04d.out' % (args.data[:7], args.infName, args.jobname, taskID)
  errname = 'logs/%s/%s/%s-%04d.err' % (args.data[:7], args.infName, args.jobname, taskID)
  print 'all future IO directed to ', outname
  outfile = MyLogFile( outname )
  errfile = MyLogFile( errname )


CMDlist = [ pyCMD, 'LearnGMM.py']
CMDlist.extend( sys.argv[1:]  ) 
CMDlist.extend( ['--taskid', '%d' % (taskID)] )

print 'Launching subprocess...'
print ' '.join( CMDlist )

if sys.stdout.isatty():
  myproc = subprocess.Popen( CMDlist )
  myproc.wait()
else:
  sys.stdout.flush()
  myproc = subprocess.Popen( CMDlist, bufsize=0, stdout=subprocess.PIPE, stderr=errfile )
  for line in iter(myproc.stdout.readline,''):
    outfile.write( line )
  myproc.wait()
