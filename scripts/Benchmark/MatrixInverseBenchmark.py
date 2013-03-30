import numpy as np
import scipy.linalg
import time
import commands
import argparse

Krange = [10, 20, 40, 80, 160]
    
def run_invcholS_dot_X( M=100, nTrial=10, Krange=Krange, doVerbose=False):
  if doVerbose: print "Benchmark: inv(chol(S)) * X"
  for K in Krange:
    S = np.random.rand( K,K)
    S = np.dot(S.T,S)
    cholS = scipy.linalg.cholesky( S, lower=True)
    X = np.random.rand( K,M)

    tstart = time.time()
    for rep in xrange( nTrial):
      np.linalg.solve( cholS, X)
    elapsedtime = time.time()-tstart

    if doVerbose:
      print "  %d x %d x %6d | %.3f sec/trial" % (K, K, M, elapsedtime/nTrial )
    else:
      print "%.3f" % (elapsedtime/nTrial)

def run_invS_dot_X( M=1000, nTrial=10, Krange=Krange, doVerbose=False):
  if doVerbose: print "Benchmark: inv(S)* X"
  for K in Krange:
    S = np.random.rand( K, K )
    S = np.dot( S.T, S)
    X = np.random.rand( K, M )
        
    tstart = time.time()
    for rep in xrange(nTrial):
      np.linalg.solve( S, X)
    elapsedtime = time.time()-tstart
  
    if doVerbose:
      print "  %d x %d x %6d | %.3f sec/trial" % (K, K, M, elapsedtime/nTrial )
    else:
      print "%.3f" % ( elapsedtime/nTrial )

def run_invS_dot_XT( M=1000,nTrial=10, Krange=Krange, doVerbose=False):
  if doVerbose: print "Benchmark: inv(S)*X' "
  for K in Krange:
    S = np.random.rand( K,K)
    S = np.dot( S.T, S)
    X = np.random.rand(M,K)
    tstart=time.time()
    for rep in xrange(nTrial):
      np.linalg.solve( S, X.T)
    elapsedtime = time.time()-tstart

    if doVerbose:
      print "  %d x %d %6d | %.3f sec/trial" % (K,K,M,elapsedtime/nTrial)
    else:
      print "%.3f" % (elapsedtime/nTrial)


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
  parser.add_argument( 'testName', type=str, default='invSX' )
  parser.add_argument( '-v', '--doVerbose', action='store_true', default=False )
  parser.add_argument( '--doBoth', action='store_true', default=False )
  parser.add_argument( '--doSingleThread', action='store_true', default=False )
  args = parser.parse_args()

  print '#------------- Python'
  if args.testName == 'invSX':
    run_invS_dot_X( M=2.5e5, Krange=Krange, doVerbose=args.doVerbose)
  elif args.testName == 'invSXT':
    run_invS_dot_XT( M=2.5e5, Krange=Krange, doVerbose=args.doVerbose)
  elif args.testName == 'invcholSX':
    run_invcholS_dot_X( M=2.5e5, Krange=Krange, doVerbose=args.doVerbose)

  MLABCMD =  "MatrixInverseBenchmark('%s', %d)"%(args.testName, args.doVerbose)
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
  
