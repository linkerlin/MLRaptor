from ..data import ToyData
from ..obsModel.GaussWishDistr import GaussWishDistr
from ..mix.QDPGMM import QDPGMM
from ..learn.VBLearnAlg import VBLearnAlg

import numpy as np
import time

def demoQGMM():
  np.random.seed( hash(time.time()) % 10000 )
  seed = np.random.randint(100000)
  print 'SEED:', seed
  print '------------'

  gw = GaussWishDistr( D=5 )

  q = QDPGMM( K=10, alpha0=0.5, obsPrior=gw)

  vb = VBLearnAlg( q, printEvery=50, nIter=500, initname='random')
 
  Data =ToyData.get_data(seed=seed)
  LP = vb.fit( Data, seed=seed )
  
  np.set_printoptions( linewidth=120, precision=2, suppress=True)
  print np.exp( q.Elogw )

if __name__ == '__main__':
  demoQGMM()
