'''
EasyToyBernData

  Streaming data generator that draws samples from a simple 
     Bernoulli mixture model with 3 easy-to-separate components
     
  Author: Mike Hughes (mike@michaelchughes.com)
'''
import numpy as np
from collections import defaultdict
np.set_printoptions( precision=2, suppress=True)

K = 10
V = 25
D = 1500
Nperdoc = 100
alpha = 0.1
beta  = 0.05 #125

w = np.ones( K )
w /= w.sum()

Phi = beta*np.ones( (K,V) )
sqrtV = int( np.sqrt(V) )
for kk in xrange( K/2):
  Phi[kk, kk*sqrtV+np.arange(sqrtV) ] = 1.0

for kk in xrange( K/2):
  colRange = kk + np.arange(0,V,sqrtV)
  Phi[sqrtV+kk, colRange] = 1.0

Phi /= Phi.sum(axis=1)[:,np.newaxis]

def sample_data_as_mat():
  DocTermMat = np.zeros( (D,V) )
  TrueW = np.zeros( (D,K) )
  for docID in xrange(D):
    zID = np.where( np.random.multinomial( 1.0, w) )[0][0]
    DocTermMat[docID,:] = np.random.multinomial( Nperdoc, Phi[zID] )
  return DocTermMat, TrueW

def sample_data_as_dict():
  BoW = list()
  nObs = 0
  GroupIDs = list()
  nTotalEntry = 0
  global TrueW
  TrueW = np.zeros( (D,K) )
  for docID in xrange( D ):
    w = np.random.dirichlet( alpha*np.ones(K) )
    Npercomp = np.random.multinomial( Nperdoc, w)
    docDict = defaultdict( int )
    for k in xrange(K):
      wordCounts =np.random.multinomial(  Npercomp[k], Phi[k] )
      for (wordID,count) in enumerate(wordCounts):
        if count == 0: 
          continue
        docDict[wordID] += count
        nObs += count
    nDistinctEntry = len(docDict )
    GroupIDs.append( (nTotalEntry,nTotalEntry+nDistinctEntry) )  
    nTotalEntry += nDistinctEntry
    BoW.append( docDict)
    TrueW[docID] = w
  return BoW, nObs, nTotalEntry, GroupIDs, TrueW
  
def print_data_info( modelName ):
  print 'Easy-to-learn toy data for K=3 Bernoulli Obs Model'
  print '  Mix weights:  '
  if modelName.count( 'Admix')>0:
    for rowID in xrange( 3):
      print '                ', np2flatstr( TrueW[rowID], '%3.2f' )
  else:
      print '                ', np2flatstr( w, '%3.2f' )
  print '  Topic-word Probs:  '
  for k in range( K ):
    print '                ', np2flatstr( Phi[k], '%4.2f' )

#######################################################################  Mixture data
def get_data( seed=8675309, **kwargs ):
  if seed is not None:
    np.random.seed( seed )
  X,TrueW= sample_data_as_mat()
  Data = dict( X=X, nObs=X.shape[0], nVocab=V, TrueW=TrueW)
  return Data


#######################################################################  Admixture data
def get_data_by_groups( seed=8675309, **kwargs ):
  if seed is not None:
    np.random.seed( seed )
  BoW, nObs, nEntry, GroupIDs,TrueW= sample_data_as_dict()
  wordidvec = np.hstack( [np.asarray(docDict.keys()) for docDict in BoW] )
  countvec = np.hstack( [np.asarray(docDict.values()) for docDict in BoW] )
  Data = dict( BoW=BoW, nObsEntry=nEntry, nObs=nObs, nDoc=D, nVocab=V, GroupIDs=GroupIDs, nGroup=len(GroupIDs), countvec=countvec, TrueW=TrueW)
  return Data

def minibatch_generator(  batch_size=1000, nBatch=50, nRep=1, seed=8675309, **kwargs):
  for repID in range( nRep ):
    np.random.seed( seed )
    for batchID in range( nBatch ):
      pass

def np2flatstr( X, fmt='% 7.2f' ):
  return ' '.join( [fmt % x for x in X.flatten() ] )  

