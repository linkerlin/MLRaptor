import numpy as np
import time
import commands

Krange = [10, 20, 40, 80, 160]
BigKrange = [1e4, 2e4,  4e4,  8e4, 16e4]

def runXTX( N=2e5, nTrial=10 ):
  print "Benchmark: X^T * X"
  for K in Krange:
    X = np.random.rand( N, K )
  
    tstart = time.time()
    for rep in xrange(nTrial):
      np.dot( X.T, X)
    elapsedtime = time.time()-tstart
  
    print "  %d x %4d | %.3f sec/trial" % (N, K, elapsedtime/nTrial )
    
def runXY( N=2000, M=1000, nTrial=10, Krange=Krange):
  print "Benchmark: X * Y"
  for K in Krange:
    X = np.random.rand( N, K)
    Y = np.random.rand( K, M )
    
    tstart = time.time()
    for rep in xrange(nTrial):
      np.dot( X, Y)
    elapsedtime = time.time()-tstart
  
    print "  %d x %6d x %d | %.3f sec/trial" % (N, K, M, elapsedtime/nTrial )
        
def runMATLABScript( MLABCMD, doSingleThread=False ):
  if doSingleThread:
    CMD = 'matlab -nodesktop -nosplash -singleCompThread -r "%s; exit;"' % (MLABCMD)
  else:
    CMD = 'matlab -nodesktop -nosplash -r "%s; exit;"' % (MLABCMD)
  print "    ", CMD
  status, stdout = commands.getstatusoutput( CMD )
  status, out = commands.getstatusoutput( 'stty sane' )
  stdout = stdout.split("www.mathworks.com.")[1]
  for line in stdout.split("\n"):
    if len( line.strip() ) > 0:
      print line   
        
if __name__ == '__main__':

  print '-------------- Python'
  runXY(N=100, M=100, Krange=BigKrange)
  runXTX()
  print '-------------- MATLAB (normal)'
  runMATLABScript( "MatrixProductBenchmark" )
  
  print '-------------- MATLAB (single thread)'
  runMATLABScript( "MatrixProductBenchmark", doSingleThread=True )
  