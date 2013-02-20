from ..data import ToyData
from ..obsModel.GaussDistr import GaussDistr
from ..mix.GMM import GMM
from ..learn.EMLearnAlg import EMLearnAlg

def demoEMGMM():
  gmm = GMM( K=3, alpha0=0.5)

  em = EMLearnAlg( gmm, printEvery=1)
 
  Data = {'X': ToyData.get_data()}
  print Data
  em.fit( Data, seed=42 )
  
if __name__ == '__main__':
  demoEMGMM()
