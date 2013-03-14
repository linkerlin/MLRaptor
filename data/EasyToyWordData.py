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
alpha = 0.5
beta  = 0.05

w = np.ones( K )
w /= w.sum()

phi = np.zeros( (K,V) )
phi[0,:] = [ 1, 1, 1, 0, 0, 0, 0, 0, 0]
phi[1,:] = [ 0, 0, 0, 1, 1, 1, 0, 0, 0]
phi[2,:] = [ 0, 0, 0, 0, 0, 0, 1, 1, 1]
phi[3,:] = [ 1, 0, 0, 1, 0, 0, 1, 0, 0]
phi[4,:] = [ 0, 1, 0, 0, 1, 0, 0, 1, 0]
phi[5,:] = [ 0, 0, 1, 0, 0, 1, 0, 0, 1]
phi += beta
for k in xrange(K):
  phi[k,:] /= np.sum( phi[k,:] )

def sample_data_as_dict():
  BoW = list()
  nObs = 0
  GroupIDs = list()
  for docID in xrange( D ):
    w = np.random.dirichlet( alpha*np.ones(K) )
    Npercomp = np.random.multinomial( Nperdoc, w)
    docDict = defaultdict( int )
    nStart = nObs
    for k in range(K):
      wordCounts =np.random.multinomial(  Npercomp[k], phi[k] )
      for (wordID,count) in enumerate( wordCounts):
        docDict[wordID] += count
        nObs += count
    GroupIDs.append( (nStart,nObs) )  
    BoW.append( docDict)
  return BoW, nObs, GroupIDs

def sample_data_as_matrix( Npercomp ):
  X = np.zeros( (Npercomp.sum(), V) )  
  for k in range(K):
    wordCounts =np.random.multinomial(  Npercomp[k], phi[k] )
    for (vv,count) in enumerate( wordCounts):
      X[ rowID, vv] = count
  return {'X':X, 'nObs':X.shape[0]}

def sample_data_from_comp( k, Nk ):
  return np.random.multinomial( Nk, phi[k] )
  
def print_data_info( modelName ):
  print 'Easy-to-learn toy data for K=3 Bernoulli Obs Model'
  print '  Mix weights:  '
  print '                ', np2flatstr( w )
  print '  Bern Probs:  '
  for k in range( K ):
    print '                ', np2flatstr( Phi[k] )

#######################################################################  Mixture data
def get_data_by_groups( seed=8675309, **kwargs ):
  BoW, nObs, GroupIDs = sample_data_as_dict()
  Data = dict( BoW=BoW, nObs=nObs, nDoc=D, nVocab=V, GroupIDs=GroupIDs )
  return Data

def minibatch_generator(  batch_size=1000, nBatch=50, nRep=1, seed=8675309, **kwargs):
  for repID in range( nRep ):
    np.random.seed( seed )
    for batchID in range( nBatch ):
      pass    

def np2flatstr( X, fmt='% 7.2f' ):
  return ' '.join( [fmt % x for x in X.flatten() ] )  

