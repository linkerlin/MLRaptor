import argparse
import numpy as np
import time

import SerialGMMLogPDF as Serial
import GlobalSharedGMMLogPDF as Global
import SharedMemGMMLogPDF as ShMem
import CyParallelGMMLogPDF as CyPar


#================================================================== Experiment Details
def create_data( N, K, D):
  X = 10*np.random.randn( N, D )
  w = np.ones( K )/float(K)  
  MuList = 5*np.random.randn( K, D)
  SigmaList = .25*np.random.rand( K, D, D)
  for k in xrange(K):
    SigmaList[k] += np.eye(D)
    SigmaList[k] = np.dot( SigmaList[k].T, SigmaList[k] )
  return X, w, MuList, SigmaList  

def run_experiment( args ):
  X, w, MuList, SigmaList = create_data( args.N, args.K, args.D)
  
  # Execute algorithm
  print "Running GMM Estep with %d processes" % (args.nProc)
  stime = time.time()
  for trial in xrange( args.nTrial ):
    if args.nProc == 1:
      logResp = Serial.Estep_serial( X, w, MuList, SigmaList )
    else:
      if args.method == 'shmem':
        logResp = ShMem.Estep_parallel( X, w, MuList, SigmaList, nProc=args.nProc, chunksize=args.chunksize, doVerbose=args.doVerbose )
      elif args.method == 'global':
        logResp = Global.Estep_parallel( X, w, MuList, SigmaList, nProc=args.nProc, chunksize=args.chunksize, doVerbose=args.doVerbose )
      elif args.method == 'cython':
        logResp = CyPar.Estep_parallel( X, w, MuList, SigmaList )
  etime = time.time() - stime
  print "%.3f sec after %d trials" % (etime, args.nTrial)
  print logResp

#================================================================== Main Entrypoint
if __name__ == '__main__':
  np.random.seed( 8675309)
  Parser = argparse.ArgumentParser()

  Parser.add_argument( '--nTrial', type=int, default=1)
  Parser.add_argument( '--nProc', type=int, default=1)
  Parser.add_argument( '--chunksize', type=int, default=1)
  Parser.add_argument( '-v', '--doVerbose', action='store_true', default=False)
  Parser.add_argument( '--method', type=str, default='global')
  Parser.add_argument( '--K', type=int, default=25)
  Parser.add_argument( '--D', type=int, default=64)
  Parser.add_argument( '--N', type=int, default=250000)
  args=Parser.parse_args()

  run_experiment( args )

