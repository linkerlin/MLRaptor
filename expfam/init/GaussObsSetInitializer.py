
import numpy as np
import scipy.spatial
import scipy.cluster
from ..util.MLUtil import logsumexp
from ..hmm.HMMUtil import FwdBwdAlg

class GaussObsSetInitializer( object ):
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
    LP['resp'],dummy = self.init_resp( Data['X'], expfamModel.K )
    SS = expfamModel.get_global_suff_stats( Data, LP )
    expfamModel.update_global_params( SS )

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
    

  '''
  def init_params_perGroup(self, Data, LP ):
    GroupIDs = Data['GroupIDs']
    LP['N_perGroup'] = np.zeros( (len(GroupIDs),self.expfamModel.K)  )
    for gg in range( len(GroupIDs) ):
      LP['N_perGroup'][gg] = np.sum( LP['resp'][ GroupIDs[gg] ], axis=0 )
    return LP
  '''
##################################################### Initialization methods
  def init_resp( self, X, K ):
    '''Initialize cluster responsibility matrix given data matrix X.

      Returns
      -------
        resp : N x K array
                  resp[n,k] = posterior prob. that item n belongs to cluster k
    '''
    if self.initname == 'kmeans':
      resp,lpr = self.get_kmeans_resp( X, K )
    elif self.initname == 'random':
      resp,lpr = self.get_random_resp( X, K )
    return resp, lpr

  def get_kmeans_resp( self, X, K ):
    ''' Kmeans initialization of cluster responsibilities.
          We run K-means algorithm on NxD data X
             and return the posterior membership probabilities to K cluster ctrs
        Params
        -------
          X      : N x D array
          K      : integer number of clusters
          doHard : boolean flag,
                       true  -> resp[n,:] is a one-hot vector,
                       false -> resp[n,:] contains actual probabilities

        Returns
        -------
          resp : N x K array
                  resp[n,k] = posterior prob. that item n belongs to cluster k
    '''
    N,D = X.shape
    scipy.random.seed( self.seed )
    Mu, Z = scipy.cluster.vq.kmeans2( X, K, iter=25, minit='points' )
    Dist = scipy.spatial.distance.cdist( X, Mu )
    resp,lpr = self.get_resp_from_distance_matrix( Dist )
    return resp, lpr

  def get_random_resp( self, X, K ):
    ''' Random sampling initialization of cluster responsibilities.
           
        Params
        -------
          X      : N x D array
          K      : integer number of clusters
          doHard : boolean flag,
                       true  -> resp[n,:] is a one-hot vector,
                       false -> resp[n,:] contains actual probabilities

        Returns
        -------
          resp : N x K array
                  resp[n,k] = posterior prob. that item n belongs to cluster k
    '''
    N,D = X.shape
    ctrIDs = self.ctrIDs
    if ctrIDs is None:  
      np.random.seed( self.seed )
      ctrIDs = np.random.permutation( N )[:K]
    else:
      assert len(ctrIDs) == K
    Dist   = scipy.spatial.distance.cdist( X, X[ctrIDs] )
    return self.get_resp_from_distance_matrix( Dist )


  def get_resp_from_distance_matrix( self, Dist ):
    ''' Get posterior probabilities given a matrix that measures
          distance from each data point to current cluster centers

        Params
        -------
          Dist : NxK matrix
                  Dist[n,k] = euclidean distance from X[n] to Mu[k]

          doHard : boolean flag,
                    true  -> resp[n,:] is a one-hot vector,
                    false -> resp[n,:] contains actual probabilities

        Returns
        -------
          resp : NxK matrix
                  resp[n,k] \propto  exp( -Dist[ X[n], Mu[k] ]  )
                  resp[n,:] sums to 1

                  if doHard, resp is a one-hot vector where
                    resp[n,k] = 1 iff Mu[k] is closest to X[n]
    '''
    N,K = Dist.shape
    if self.doHard:
      grid = np.tile( np.arange(K), (N,1) )
      resp = np.asfarray( grid == np.argmin(Dist,axis=1)[:,np.newaxis] )
    else:
      # Now make an N x K matrix resp, 
      #  where resp[n,k] \propto exp( -dist(X[n], Center[k]) )
      lpr = -Dist
      lprPerItem = logsumexp( lpr, axis=1 )
      resp  = np.exp( lpr-lprPerItem[:,np.newaxis] )
      resp /= np.sum(resp,axis=1)[:,np.newaxis]
    assert np.allclose( np.sum(resp,axis=1), 1.0 )  
    return resp, lpr
