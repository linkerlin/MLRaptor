from ..data import ToyData
from ..obsModel.GaussDistr import GaussDistr
from ..mix.GMM import GMM
from ..learn.OnlineEMLearnAlg import OnlineEMLearnAlg

import numpy as np
import time

np.set_printoptions( linewidth=120, precision=2, suppress=True)

def demoEMGMM():
  np.random.seed( hash(time.time()) % 10000 )
  seed = np.random.randint(100000)
  print 'SEED:', seed
  print '------------'
  
  gmm = GMM( K=3, alpha0=0)

  em = OnlineEMLearnAlg( gmm, printEvery=5, initname='random')
 
  DG = ToyData.minibatch_generator( nBatch=100, nRep=2, seed=seed)
  Dtest = ToyData.get_data( seed=42)
  em.fit( DG, seed=seed, Dtest=Dtest )
  
  print gmm.obsDistr[0].mu
  print gmm.obsDistr[1].mu
  print gmm.obsDistr[2].mu
  
if __name__ == '__main__':
  demoEMGMM()
