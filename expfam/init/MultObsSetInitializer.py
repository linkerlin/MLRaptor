
import numpy as np
import scipy.spatial
import scipy.cluster
from ..util.MLUtil import logsumexp
from ..hmm.HMMUtil import FwdBwdAlg

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
    LP['resp'] = self.init_resp_random( Data, expfamModel.K)
    SS = expfamModel.get_global_suff_stats( Data, LP )
    expfamModel.update_global_params( SS )

  def init_resp_random( self, Data, K):
    np.random.seed( self.seed)
    resp = np.random.rand( Data['nObsEntry'], K )
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
