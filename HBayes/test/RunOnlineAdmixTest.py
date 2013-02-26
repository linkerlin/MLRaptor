from ..data import ToyData
from ..obsModel.GaussWishDistr import GaussWishDistr
from ..admix.QGAM import QGAM
from ..learn.OnlineVBLearnAlg import OnlineVBLearnAlg

import numpy as np
import time

np.set_printoptions( linewidth=120, precision=3, suppress=True)
def demoQGMM():
  np.random.seed( hash(time.time()) % 10000 )
  seed = np.random.randint(100000)
  print 'SEED:', seed
  print '------------'

  gw = GaussWishDistr( D=5 )

  qgam = QGAM( K=3, alpha0=0.5, obsPrior=gw)

  ovb = OnlineVBLearnAlg( qgam, printEvery=1, nIter=100, initname='kmeans')
 
  DataGen =ToyData.group_minibatch_generator(seed=seed)
  Dtest = ToyData.get_data_by_groups( nGroup=4, nPerGroup=5000, seed=42 )
  LP = ovb.fit( DataGen, seed=seed, Dtest=Dtest )
  
  
  #print 'True W __ per group'
  #print Dtest['TrueW_perGroup'][-10:]
  
  #tLP = None
  #for rr in xrange(4):  
  #  tLP = qgam.calc_local_params( Dtest, tLP)
  #  print '  Learned W __ per group'
  #  print np.exp( tLP['Elogw_perGroup'] )[-10:]


if __name__ == '__main__':
  demoQGMM()
