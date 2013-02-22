from ..data import ToyData
from ..obsModel.GaussWishDistr import GaussWishDistr
from ..mix.QGMM import QGMM
from ..learn.VBLearnAlg import VBLearnAlg

import numpy as np
import time

def demoQGMM():
  gw = GaussWishDistr( D=5 )
  np.random.seed( hash(time.time()) % 10000 )
  seed = np.random.randint(100000)
  print 'SEED:', seed
  print '-----------'

  qgmm = QGMM( K=3, alpha0=0.5, obsPrior=gw)

  vb = VBLearnAlg( qgmm, printEvery=1)
 
  Data = {'X': ToyData.get_data( seed=seed)}

  vb.fit( Data, seed=seed )
  
if __name__ == '__main__':
  demoQGMM()
