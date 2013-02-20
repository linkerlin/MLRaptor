from ..data import ToyData
from ..obsModel.GaussWishDistr import GaussWishDistr
from ..mix.QGMM import QGMM
from ..learn.VBLearnAlg import VBLearnAlg

def demoQGMM():
  gw = GaussWishDistr( D=5 )

  qgmm = QGMM( K=3, alpha0=0.5, obsPrior=gw)

  vb = VBLearnAlg( qgmm, printEvery=1)
 
  Data = {'X': ToyData.get_data()}
  print Data
  vb.fit( Data, seed=42 )
  
if __name__ == '__main__':
  demoQGMM()
