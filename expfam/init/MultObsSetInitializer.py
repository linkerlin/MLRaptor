
import numpy as np
import random
import scipy.spatial
import scipy.cluster

from ..obsModel.DirichletDistr import DirichletDistr
from ..util.MLUtil import logsumexp
from ..hmm.HMMUtil import FwdBwdAlg

np.set_printoptions( precision=2, suppress=True)

class MultObsSetInitializer( object ):
  def __init__( self, initname, seed, doHard=False, ctrIDs=None):
    self.initname = initname
    self.seed = seed
    self.doHard = doHard
    self.ctrIDs = ctrIDs

  def init_global_params( self, expfamModel, Data ):
    ''' Initialize all global params (allocModel + obsModel)
          for given expfam model using this obj's specified init procedure.
        Essentially, we ignore fancy structure and pretend all observations are iid.
          and simply construct the NxK posterior responsibilities
            either using K centers chosen at random from the data,
                or using K centers learned via kmeans 
    '''
    expfamModel.set_obs_dims( Data )
    if 'nSeq' in Data:
      self.init_params_sequenceModel( expfamModel, Data)
    else:
      self.init_params_mixModel( expfamModel, Data)

  def init_params_mixModel( self, expfamModel, Data):
    LP = dict()
    if self.initname == 'cheat':
      LP['resp'] = self.init_resp_cheat( Data, expfamModel.K)
    if self.initname == 'randsample':
      LP['resp'] = self.init_resp_randsample( Data, expfamModel.K)
    elif self.initname == 'kmeans':
      LP['resp'] = self.init_resp_kmeans( Data, expfamModel.K)
    else:
      LP['resp'] = self.init_resp_random( Data, expfamModel.K)
    SS = expfamModel.get_global_suff_stats( Data, LP )
    expfamModel.update_global_params( SS )

  def init_resp_cheat(self, Data, K):
    Phi = np.zeros( (K,Data['nVocab']) )
    Phi[0,:] = [ 1, 1, 1, 0, 0, 0, 0, 0, 0]
    Phi[1,:] = [ 0, 0, 0, 1, 1, 1, 0, 0, 0]
    Phi[2,:] = [ 0, 0, 0, 0, 0, 0, 1, 1, 1]
    Phi[3,:] = [ 1, 0, 0, 1, 0, 0, 1, 0, 0]
    Phi[4,:] = [ 0, 1, 0, 0, 1, 0, 0, 1, 0]
    Phi[5,:] = [ 0, 0, 1, 0, 0, 1, 0, 0, 1]
    Phi += 0.01
    return self.calc_resp_given_topic_word_param( Data,Phi,K)
        
  def init_resp_kmeans( self, Data, K):
    '''  Select K documents via kmeans, and then use these as centers
    '''
    try:
      X = Data['X']
    except KeyError:
      X = np.zeros( (Data['nGroup'],Data['nVocab']) )
      for docID in xrange( Data['nGroup'] ):
        for wID,count in Data['BoW'][docID].items():
          X[docID,wID] = count
    scipy.random.seed( self.seed )
    Phi, Z = scipy.cluster.vq.kmeans2( X, K, iter=25, minit='points' )
    Phi += 1e-5
    return self.calc_resp_given_topic_word_param( Data,Phi,K)
  
  def init_resp_randsample( self, Data, K):
    '''  Select K documents at "random" and use these as centers
    '''
    random.seed( self.seed)
    try:
      docIDs = random.sample( xrange( Data['X'].shape[0]),K)      
      Phi = Data['X'][ docIDs,:]
      Phi += 1e-5
    except KeyError:
      docIDs = random.sample( xrange(Data['nGroup']), K )
      Phi = 0.01*np.random.rand(K, Data['nVocab'])
      for kk,docID in enumerate(docIDs):
        for wID,count in Data['BoW'][docID].items():
          Phi[kk,wID] = count
    return self.calc_resp_given_topic_word_param( Data,Phi,K)
    
  def calc_resp_given_topic_word_param( self, Data, Phi,K): 
    try:
      lpr = np.zeros( (Data['nObsEntry'],K) )
    except KeyError:
      lpr = np.zeros( (Data['nObs'],K) )
    Phi /= Phi.sum(axis=1)[:,np.newaxis]
    for kk in xrange(K):
      MyDirDistr = DirichletDistr( Phi[kk] )
      lpr[:,kk] = MyDirDistr.E_log_pdf( Data )
    lprPerItem = logsumexp( lpr, axis=1 )
    resp   = np.exp( lpr-lprPerItem[:,np.newaxis] )
    resp   /= resp.sum( axis=1)[:,np.newaxis] # row normalize
    return resp
    
  def init_resp_random( self, Data, K):
    np.random.seed( self.seed)
    try:
      resp = np.random.rand( Data['nObsEntry'], K )
    except KeyError:
      resp = np.random.rand( Data['nObs'], K )
    resp /= resp.sum(axis=1)[:,np.newaxis]
    return resp

  def init_params_sequenceModel( self, expfamModel, Data):
    '''  Obtain initial obs param estimates via kmeans or random selection,
          then run Fwd/Bwd using uniform transition matrix to get needed LP params
    '''
    K = expfamModel.K
    PiInit = 1.0/K*np.ones( K)
    PiMat = 1.0/K*np.ones( (K,K) )
    LP = dict()
    LP['resp'], LP['E_log_soft_ev'] = self.init_resp( Data['X'], K )
    LP['respPair'] = list()
    for ii in xrange( Data['nSeq'] ):
      seqLogSoftEv =  LP['E_log_soft_ev'][ Data['Tstart'][ii]:Data['Tstop'][ii] ]
      seqResp, seqRespPair, seqLogPr = FwdBwdAlg( PiInit, PiMat, seqLogSoftEv )
      LP['resp'][ Data['Tstart'][ii]:Data['Tstop'][ii] ] = seqResp        
      LP['respPair'].append( seqRespPair )       
    SS = expfamModel.get_global_suff_stats( Data, LP )
    expfamModel.update_global_params( SS )
