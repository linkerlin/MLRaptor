from ..data import ToyData
from ..obsModel.GaussWishDistr import GaussWishDistr
from ..mix.QGMM import QGMM
from ..learn.VBLearnAlg import VBLearnAlg

import numpy as np
import time

def demoQGMM():
  np.random.seed( hash(time.time()) % 10000 )
  seed = np.random.randint(100000)
  print 'SEED:', seed
  print '------------'

  gw = GaussWishDistr( D=5 )

  q = QGMM( K=10, alpha0=0.1, obsPrior=gw)

  vb = VBLearnAlg( q, printEvery=50, nIter=500, initname='random')
 
  Data =ToyData.get_data(seed=seed)
  Data = {'X':Data}
  LP = vb.fit( Data, seed=seed )
  
  np.set_printoptions( linewidth=120, precision=2, suppress=True)
  print np.exp( q.Elogw )

if __name__ == '__main__':
  demoQGMM()
