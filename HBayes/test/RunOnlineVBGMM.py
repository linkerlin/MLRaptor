from ..data import ToyData
from ..obsModel.GaussWishDistr import GaussWishDistr
from ..mix.QGMM import QGMM
from ..learn.OnlineVBLearnAlg import OnlineVBLearnAlg

import argparse
import numpy as np
import time

def demoVBGMM( seed, Ntotal, nBatch, nRep, initname):
  print 'SEED:', seed
  print '------------'
  
  gw = GaussWishDistr( D=5 )
  qgmm = QGMM( K=3, alpha0=0.5, obsPrior=gw)

  vb = OnlineVBLearnAlg( qgmm, printEvery=5, initname=initname)
 
  DG = ToyData.minibatch_generator( nBatch=nBatch, nRep=nRep, seed=seed)
  Dtest = ToyData.get_data( seed=42)
  vb.fit( DG, seed=seed, Ntotal=Ntotal, Dtest=Dtest )
  
  np.set_printoptions( linewidth=120, precision=2, suppress=True)
  print qgmm.qobsDistr[0].m
  print qgmm.qobsDistr[1].m  
  print qgmm.qobsDistr[2].m
  
if __name__ == '__main__':
  seedDefault = hash(time.time()) % 10000
  myparser = argparse.ArgumentParser()
  myparser.add_argument( '--seed', type=int, default=seedDefault)
  myparser.add_argument( '--Ntotal', type=int, default=None)
  myparser.add_argument( '--nBatch', type=int, default=100)
  myparser.add_argument( '--nRep', type=int, default=2)
  myparser.add_argument( '--initname', type=str, default='random')

  args = myparser.parse_args()
  demoVBGMM( args.seed, args.Ntotal, args.nBatch, args.nRep, args.initname)
