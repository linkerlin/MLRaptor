'''
 Abstract class for learning algorithms for Gaussian Mixture Models. 

  Simply defines some generic initialization routines.

Author: Mike Hughes (mike@michaelchughes.com)
'''
import numpy as np
import scipy.spatial
import scipy.cluster
import time

class LearnAlgGMM(object):

  def fit( self, X):
    pass

  def init_params( self):
    pass

  def init_resp( self, X, K):
    '''Initialize cluster responsibility matrix given data matrix X.

      Returns
      -------
        resp : N x K array
                  resp[n,k] = posterior prob. that item n belongs to cluster k
      
      Notes
        -------
          Relies on numpy's random number seed, which can be set with
            np.random.seed( myseed )
    '''
    if self.initname == 'kmeans':
      resp = self.get_kmeans_resp( X, K )
    elif self.initname == 'random':
      resp = self.get_random_resp( X, K )
    return resp

  ##################################################### Logging methods
  def print_state( self, iterid, evBound, doFinal=False, status=''):
    doPrint = iterid % self.printEvery==0
    logmsg  = '  %5d/%d after %6.0f sec. | evidence % .6e' % (iterid, self.Niter, time.time()-self.start_time, evBound)
    if iterid ==0:
      print 'Initialized via %s.' % (self.initname)
      print logmsg
    elif (doFinal and not doPrint):
      print logmsg
    elif (not doFinal and doPrint):
      print logmsg
    if doFinal:
      print '... done. %s' % (status)


    
  def save_state( self, iterid, evBound ):
    pass

  ##################################################### Initialization methods
  def get_kmeans_resp( self, X, K, doHard=True):
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

        Notes
        -------
          Relies on numpy's random number seed, which can be set with
            np.random.seed( myseed )
    '''
    N,D = X.shape
    Mu, Z = scipy.cluster.vq.kmeans2( X, K, iter=100, minit='points' )
    Dist = scipy.spatial.distance.cdist( X, Mu )
    return self.get_resp_from_distance_matrix( Dist, doHard )



  def get_random_resp( self, X, K, doHard=True, ctrIDs=None):
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

        Notes
        -------
          Relies on numpy's random number seed, which can be set with
            np.random.seed( myseed )
    '''
    N,D = X.shape
    if ctrIDs is None:  
      ctrIDs = np.random.permutation( N )[:K]
    else:
      assert len(ctrIDs) == K
    Dist   = scipy.spatial.distance.cdist( X, X[ctrIDs] )
    return self.get_resp_from_distance_matrix( Dist, doHard )

  def get_resp_from_distance_matrix( self, Dist, doHard):
    N,K = Dist.shape
    if doHard:
      grid = np.tile( np.arange(K), (N,1) )
      resp = np.asfarray( grid == np.argmin(Dist,axis=1)[:,np.newaxis] )
    else:
      # Now make an N x K matrix resp, 
      #  where resp[n,k] \propto exp( -dist(X[n], Center[k]) )
      logresp = -Dist + np.min(Dist,axis=1)[:,np.newaxis]
      resp   = np.exp( logresp )        
    return resp/resp.sum(axis=1)[:,np.newaxis]  
