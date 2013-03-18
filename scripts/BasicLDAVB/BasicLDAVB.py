import time
import numpy as np
from scipy.special import digamma, gammaln

alphaPrior = 0.5
lambdaPrior = 0.1

def runLDA( Data, nTopics, nIter=100, seed=324 ):
  global K
  K = nTopics
  np.random.seed( seed )
  Phi = np.random.rand( K, Data['nVocab'] )
  Phi /= Phi.sum(axis=1)[:,np.newaxis]
  #print 'INIT: ', Phi[0,:6]
  Elogphi = np.log( Phi )
  logUnifTopicDistr = np.log( 1.0/K*np.ones( K ) )
  Elogw_perDoc = [logUnifTopicDistr for docID in xrange(Data['nGroup']) ]
  tstart = time.time()
  for iterid in xrange( nIter ):
    resp = Estep_TopicAsgn( Data, Elogw_perDoc, Elogphi )
    Elogw_perDoc = Estep_DocTopicDistr( Data, resp )
    Elogphi      = Mstep( Data, resp )
    print "iter %d after %.0f sec" % (iterid, time.time()-tstart)
    #print np.exp( Elogphi[2, :10] ), '|',  Elogw_perDoc[ 77 ][:5], '|', resp[0,:7]
  return Elogphi

def Estep_DocTopicDistr( Data, resp ):
  Elogw_perDoc = [None for docID in xrange(Data['nGroup']) ]
  for docID in xrange( Data['nGroup'] ):
    docRange = xrange( Data['GroupIDs'][docID][0], Data['GroupIDs'][docID][1] )
    alphavec = alphaPrior + np.sum( resp[docRange], axis=0 )
    #if docID % 50 == 0:
    #  print np.sum( resp[docRange], axis=0 )[:5]
    Elogw_perDoc[docID] = digamma( alphavec ) - digamma( alphavec.sum() )
  return Elogw_perDoc

def Estep_TopicAsgn( Data, Elogw_perDoc, Elogphi):
  lpr = np.dot( Data['X'], Elogphi.T )  
  for docID in xrange( Data['nGroup'] ):
    docRange = xrange( Data['GroupIDs'][docID][0], Data['GroupIDs'][docID][1] )
    lpr[ docRange ] += Elogw_perDoc[docID]
  lprMAX = lpr.max(axis=1)
  lpr = lpr - lprMAX[:,np.newaxis]
  lprPerItem = np.log( np.sum( np.exp(lpr),axis=1) ) + lprMAX
  resp   = np.exp( lpr-lprPerItem[:,np.newaxis] )
  resp   /= resp.sum( axis=1)[:,np.newaxis] # row normalize
  return resp

def Mstep( Data, resp ):
  TermCount = np.dot( resp.T, Data['X'] )
  TopicTermLam = lambdaPrior + TermCount
  Elogphi = digamma( TopicTermLam ) - digamma( TopicTermLam.sum(axis=1) )[:,np.newaxis]
  return Elogphi
