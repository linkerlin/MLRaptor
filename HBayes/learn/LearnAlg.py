'''
 Abstract class for learning algorithms for Gaussian Mixture Models. 

  Simply defines some generic initialization routines, based on 
     assigning cluster responsibilities (either hard or soft) to 
     cluster centers either learned from data (kmeans)
                         or selected at random from the data

Author: Mike Hughes (mike@michaelchughes.com)
'''
import numpy as np
import scipy.spatial
import scipy.cluster
import time

class LearnAlg(object):

  def __init__( self, savefilename='results/GMMtrace', nIter=100, \
                    initname='kmeans',  convTHR=1e-10, \
                    printEvery=5, saveEvery=5, doVerify=False, \
                    **kwargs ):
    self.savefilename = savefilename
    self.initname = initname
    self.convTHR = convTHR
    self.Niter = nIter
    self.printEvery = printEvery
    self.saveEvery = saveEvery
    self.SavedIters = dict()
    self.doVerify = doVerify

  def init_params( self, Data):
    pass

  def fit( self, Data):
    pass

  def save_state( self, iterid, evBound ):
    pass

  ##################################################### Logging methods
  def verify_evidence(self, evBound, prevBound):
    isValid = prevBound < evBound or np.allclose( prevBound, evBound, rtol=self.convTHR )
    if not isValid:
      print 'WARNING: evidence decreased!'
      print '    prev = % .15e' % (prevBound)
      print '     cur = % .15e' % (evBound)
    isConverged = np.abs(evBound-prevBound)/np.abs(evBound) <= self.convTHR
    return isConverged 

  def print_state( self, iterid, evBound, doFinal=False, status=''):
    doPrint = iterid % self.printEvery==0
    logmsg  = '  %5d/%s after %6.0f sec. | evidence % .9e' % (iterid, str(self.Niter), time.time()-self.start_time, evBound)
    if iterid ==0 and not doFinal:
      print '  Initialized via %s.' % (self.initname)
      print logmsg
    elif (doFinal and not doPrint):
      print logmsg
    elif (not doFinal and doPrint):
      print logmsg
    if doFinal:
      print '... done. %s' % (status)

  ##################################################### Initialization methods
  def init_resp( self, X, K, **kwargs):
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
      resp = self.get_kmeans_resp( X, K, **kwargs )
    elif self.initname == 'random':
      resp = self.get_random_resp( X, K, **kwargs )
    return resp

  def get_kmeans_resp( self, X, K, doHard=False, seed=42):
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
    scipy.random.seed( seed )
    Mu, Z = scipy.cluster.vq.kmeans2( X, K, iter=100, minit='points' )
    Dist = scipy.spatial.distance.cdist( X, Mu )
    resp = self.get_resp_from_distance_matrix( Dist, doHard )
    return resp

  def get_random_resp( self, X, K, doHard=False, ctrIDs=None, seed=42):
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
      np.random.seed( seed )
      ctrIDs = np.random.permutation( N )[:K]
    else:
      assert len(ctrIDs) == K
    Dist   = scipy.spatial.distance.cdist( X, X[ctrIDs] )
    return self.get_resp_from_distance_matrix( Dist, doHard )


  def get_resp_from_distance_matrix( self, Dist, doHard):
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
    if doHard:
      grid = np.tile( np.arange(K), (N,1) )
      resp = np.asfarray( grid == np.argmin(Dist,axis=1)[:,np.newaxis] )
    else:
      # Now make an N x K matrix resp, 
      #  where resp[n,k] \propto exp( -dist(X[n], Center[k]) )
      logresp = -Dist + np.min(Dist,axis=1)[:,np.newaxis]
      resp   = np.exp( logresp )        
    return resp/resp.sum(axis=1)[:,np.newaxis]  
