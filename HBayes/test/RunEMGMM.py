from ..data import ToyData
from ..obsModel.GaussDistr import GaussDistr
from ..mix.GMM import GMM
from ..learn.EMLearnAlg import EMLearnAlg

import numpy as np

np.set_printoptions( linewidth=120, precision=2, suppress=False)

def demoEMGMM():
  gmm = GMM( K=3, alpha0=0.5)

  em = EMLearnAlg( gmm, printEvery=1)
 
  Data = ToyData.get_data()
  print Data
  em.fit( Data, seed=42 )
  
  print gmm.obsDistr[0].mu
  print gmm.obsDistr[1].mu  
  print gmm.obsDistr[2].mu
  
if __name__ == '__main__':
  demoEMGMM()
