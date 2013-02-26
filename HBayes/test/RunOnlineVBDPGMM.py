from ..data import ToyData
from ..obsModel.GaussWishDistr import GaussWishDistr
from ..mix.QDPGMM import QDPGMM
from ..learn.OnlineVBLearnAlg import OnlineVBLearnAlg

import numpy as np
import time

np.set_printoptions( linewidth=120, precision=2, suppress=True)

def demoDPGMM():
  np.random.seed( hash(time.time()) % 10000 )
  seed = np.random.randint(100000)
  print 'SEED:', seed
  print '------------'
  
  gw = GaussWishDistr( D=5 )
  qdp = QDPGMM( K=3, alpha0=0.5, obsPrior=gw)

  vb = OnlineVBLearnAlg( qdp, printEvery=5, initname='random')
 
  DG = ToyData.minibatch_generator( nBatch=1000, nRep=2, seed=seed)
  Dtest = ToyData.get_data( seed=42)
  vb.fit( DG, seed=seed, Dtest=Dtest )
  
  print qdp.qobsDistr[0].m
  print qdp.qobsDistr[1].m  
  print qdp.qobsDistr[2].m
  
if __name__ == '__main__':
  demoDPGMM()
