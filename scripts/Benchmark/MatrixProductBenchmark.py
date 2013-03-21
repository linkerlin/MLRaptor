import numpy as np
import time
import commands
import argparse

Krange = [10, 20, 40, 80, 160]
BigKrange = [1e4, 2e4,  4e4,  8e4, 16e4]

def runXTX( N=2e5, nTrial=10 , doVerbose=False):
  if doVerbose: print "Benchmark: X^T * X"
  for K in Krange:
    X = np.random.rand( N, K )
  
    tstart = time.time()
    for rep in xrange(nTrial):
      np.dot( X.T, X)
    elapsedtime = time.time()-tstart
  
    if doVerbose:
      print "  %d x %4d | %.3f sec/trial" % (N, K, elapsedtime/nTrial )
    else:
      print "%.3f" % ( elapsedtime/nTrial )
    
def runXTY( N=2000, M=1000, nTrial=10, Krange=Krange, doVerbose=False):
  if doVerbose: print "Benchmark: X^T * Y"
  for K in Krange:
    X = np.random.rand( K, N )
    Y = np.random.rand( K, M )
    
    tstart = time.time()
    for rep in xrange(nTrial):
      np.dot( X.T, Y)
    elapsedtime = time.time()-tstart
  
    if doVerbose:
      print "  %d x %6d x %d | %.3f sec/trial" % (N, K, M, elapsedtime/nTrial )
    else:
      print "%.3f" % ( elapsedtime/nTrial )

def runXY( N=2000, M=1000, nTrial=10, Krange=Krange, doVerbose=False):
  if doVerbose: print "Benchmark: X * Y"
  for K in Krange:
    X = np.random.rand( N, K)
    Y = np.random.rand( K, M )
    
    tstart = time.time()
    for rep in xrange(nTrial):
      np.dot( X, Y)
    elapsedtime = time.time()-tstart
  
    if doVerbose:
      print "  %d x %6d x %d | %.3f sec/trial" % (N, K, M, elapsedtime/nTrial )
    else:
      print "%.3f" % ( elapsedtime/nTrial ) 

def runMATLABScript( MLABCMD, doSingleThread=False ):
  if doSingleThread:
    CMD = 'matlab -nodesktop -nosplash -singleCompThread -r "%s; exit;"' % (MLABCMD)
  else:
    CMD = 'matlab -nodesktop -nosplash -r "%s; exit;"' % (MLABCMD)
  #print "    ", CMD
  status, stdout = commands.getstatusoutput( CMD )
  status, out = commands.getstatusoutput( 'stty sane' )
  stdout = stdout.split("www.mathworks.com.")[1]
  for line in stdout.split("\n"):
    if len( line.strip() ) > 0:
      print line   
        
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument( 'testName', type=str, default='XTX' )
  parser.add_argument( '-v', '--doVerbose', action='store_true', default=False )
  parser.add_argument( '--doBoth', action='store_true', default=False )
  parser.add_argument( '--doSingleThread', action='store_true', default=False )
  args = parser.parse_args()

  print '#------------- Python'
  if args.testName == 'XY':
    runXY(N=100, M=100, Krange=BigKrange, doVerbose=args.doVerbose)
  elif args.testName == 'XTY':
    runXTY(N=100, M=100, Krange=BigKrange, doVerbose=args.doVerbose)
  else:
    runXTX(doVerbose=args.doVerbose)

  MLABCMD =  "MatrixProductBenchmark('%s', %d)"%(args.testName, args.doVerbose)
  if args.doSingleThread:
    print '#------------- MATLAB (single thread)'
    runMATLABScript( MLABCMD, doSingleThread=True )
  elif args.doBoth:
    print '#------------- MATLAB (single thread)'
    runMATLABScript( MLABCMD, doSingleThread=True )
    print '#------------- MATLAB (multi thread)'
    runMATLABScript( MLABCMD )
  else:
    print '#------------- MATLAB (normal multi-thread)'
    runMATLABScript( MLABCMD )
