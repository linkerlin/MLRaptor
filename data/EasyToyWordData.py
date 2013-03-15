'''
EasyToyBernData

  Streaming data generator that draws samples from a simple 
     Bernoulli mixture model with 3 easy-to-separate components
     
  Author: Mike Hughes (mike@michaelchughes.com)
'''
import numpy as np
from collections import defaultdict

K = 6
V = 9
D = 100
Nperdoc = 75
alpha = 0.1
beta  = 0.02

w = np.ones( K )
w /= w.sum()

Phi = np.zeros( (K,V) )
Phi[0,:] = [ 1, 1, 1, 0, 0, 0, 0, 0, 0]
Phi[1,:] = [ 0, 0, 0, 1, 1, 1, 0, 0, 0]
Phi[2,:] = [ 0, 0, 0, 0, 0, 0, 1, 1, 1]
Phi[3,:] = [ 1, 0, 0, 1, 0, 0, 1, 0, 0]
Phi[4,:] = [ 0, 1, 0, 0, 1, 0, 0, 1, 0]
Phi[5,:] = [ 0, 0, 1, 0, 0, 1, 0, 0, 1]
Phi += beta
for k in xrange(K):
  Phi[k,:] /= np.sum( Phi[k,:] )

def sample_data_as_dict():
  BoW = list()
  nObs = 0
  GroupIDs = list()
  nTotalEntry = 0
  for docID in xrange( D ):
    w = np.random.dirichlet( alpha*np.ones(K) )
    Npercomp = np.random.multinomial( Nperdoc, w)
    docDict = defaultdict( int )
    for k in range(K):
      wordCounts =np.random.multinomial(  Npercomp[k], Phi[k] )
      for (wordID,count) in enumerate( wordCounts):
        if count == 0: 
          continue
        docDict[wordID] += count
        nObs += count
    nDistinctEntry = len(docDict )
    GroupIDs.append( (nTotalEntry,nTotalEntry+nDistinctEntry) )  
    nTotalEntry += nDistinctEntry
    BoW.append( docDict)
  return BoW, nObs, nTotalEntry, GroupIDs

def sample_data_as_matrix( Npercomp ):
  X = np.zeros( (Npercomp.sum(), V) )  
  for k in range(K):
    wordCounts =np.random.multinomial(  Npercomp[k], Phi[k] )
    for (vv,count) in enumerate( wordCounts):
      X[ rowID, vv] = count
  return {'X':X, 'nObs':X.shape[0]}

def sample_data_from_comp( k, Nk ):
  return np.random.multinomial( Nk, Phi[k] )
  
def print_data_info( modelName ):
  print 'Easy-to-learn toy data for K=3 Bernoulli Obs Model'
  print '  Mix weights:  '
  print '                ', np2flatstr( w )
  print '  Topic-word Probs:  '
  for k in range( K ):
    print '                ', np2flatstr( Phi[k] )

#######################################################################  Mixture data
def get_data_by_groups( seed=8675309, **kwargs ):
  BoW, nObs, nEntry, GroupIDs = sample_data_as_dict()
  Data = dict( BoW=BoW, nObsEntry=nEntry, nObs=nObs, nDoc=D, nVocab=V, GroupIDs=GroupIDs, nGroup=len(GroupIDs) )
  return Data

def minibatch_generator(  batch_size=1000, nBatch=50, nRep=1, seed=8675309, **kwargs):
  for repID in range( nRep ):
    np.random.seed( seed )
    for batchID in range( nBatch ):
      pass    

def np2flatstr( X, fmt='% 7.2f' ):
  return ' '.join( [fmt % x for x in X.flatten() ] )  

