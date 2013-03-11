'''
  Represents mean-field factorization of a 
    Bayesian infinite admixture model (via Hier. Dirichlet Process)
    automatically determines number of components

 Author: Mike Hughes (mike@michaelchughes.com)

 Parameters
 -------
   alpha0  : scalar hyperparameter of global stick-breaking GEM prior on mix. weights
   gamma   : scalar hyperparameter of group-level Dirichlet prior on mix. weights
    
 Usage
 -------

 Inference
 -------
'''

import numpy as np
from scipy.special import gammaln, digamma
from ..util.MLUtil import logsumexp, np2flatstr, flatstr2np

import GlobalStickPropOptimizer

EPS = 10*np.finfo(float).eps

class QHDPAdmixModel( object ):

  def __init__( self, K, alpha0, gamma=1, truncType='z', **kwargs):
    self.qType = 'VB'
    self.K = K
    self.alpha1 = 1.0
    self.alpha0 = alpha0    
    self.gamma  = gamma
    self.truncType = truncType

    #  q( v_k ) = point estimate
    self.vstar = self.alpha1/(self.alpha1+self.alpha0) *np.ones( K )
    self.Ebeta = self.get_beta()

  def get_info_string( self):
    return 'HDP infinite admixture model with %d components' % (self.K)
        
  def to_string( self ):
    return np2flatstr( self.vstar )

  def calc_local_params( self, Data, LP):
    ''' E-step
          alternate between these updates until convergence
             q(Z)  (posterior on topic-token assignment)
         and q(W)  (posterior on group-topic distribution)
    '''
    try:
      LP['N_perGroup']
    except KeyError:
      LP['N_perGroup'] = np.zeros( (Data['nGroup'],self.K+1) )

    GroupIDs = Data['GroupIDs']
    nGroups = Data['nGroup']
    prevVec = None
    for rep in xrange( 10 ):
      LP = self.get_local_W_params( Data, LP)
      LP = self.get_local_Z_params( Data, LP)
      for gg in range( nGroups ):
        LP['N_perGroup'][gg,:-1] = np.sum( LP['resp'][ GroupIDs[gg] ], axis=0 )
      curVec = LP['alpha_perGroup'].flatten()
      if prevVec is not None and np.allclose( prevVec, curVec ):
        break
      prevVec = curVec
    return LP
    
  def get_local_W_params( self, Data, LP):
    GroupIDs = Data['GroupIDs']
    alpha_perGroup = self.gamma*self.Ebeta + LP['N_perGroup']

    LP['alpha_perGroup'] = alpha_perGroup
    LP['Elogw_perGroup'] = digamma( alpha_perGroup ) \
                             - digamma( alpha_perGroup.sum(axis=1) )[:,np.newaxis]
    return LP
    
  def get_local_Z_params( self, Data, LP):
    GroupIDs = Data['GroupIDs']
    lpr = LP['E_log_soft_ev'].copy() # need copy so we can do += later
    for gg in xrange( len(GroupIDs) ):
      lpr[ GroupIDs[gg] ] += LP['Elogw_perGroup'][gg, :-1]
    lprPerItem = logsumexp( lpr, axis=1 )
    resp   = np.exp( lpr-lprPerItem[:,np.newaxis] )
    resp   /= resp.sum( axis=1)[:,np.newaxis] # row normalize
    assert np.allclose( resp.sum(axis=1), 1)
    LP['resp'] = resp
    return LP

  def get_global_suff_stats( self, Data, SS, LP ):
    ''' 
    '''
    SS = dict()
    SS['N'] = np.sum( LP['resp'], axis=0 )
    SS['Nall'] = SS['N'].sum()
    try:
      SS['Elogw'] = np.sum( LP['Elogw_perGroup'], axis=0 )
    except KeyError:
      SS['Elogw'] = self.gamma*self.Ebeta
    SS['G'] = Data['nGroup']
    
    '''
    print Data['TrueW_perGroup'][:4,:]
    print '---'
    try:
      print np.exp( LP['Elogw_perGroup'][:4,:] )
    except KeyError:
      pass
    '''  
    return SS
    
  def update_global_params( self, SS, rho=None, Ntotal=None, **kwargs ):
    ''' Run optimization to find best global stick-proportions v
    '''
    ampF = 1 #no matter what!
    args = ( SS['G'], SS['Elogw'], self.alpha0, self.gamma)
    vstar = GlobalStickPropOptimizer.get_best_stick_prop_point_est(self.K, *args, vinitIN=self.vstar)
    if rho is None or rho == 1:
      self.vstar = vstar
    else:
      self.vstar = rho*vstar + (1-rho)*self.vstar
    self.Ebeta = self.get_beta()
    print self.Ebeta
    
  def get_beta( self):
    v = np.hstack( [self.vstar, 1] )
    c1mv = np.cumprod( 1 - v )
    c1mv = np.hstack( [1, c1mv] )
    beta = v * c1mv[:-1]
    return beta

  def calc_evidence( self, Data, SS, LP ):
    GroupIDs = Data['GroupIDs']
    return self.E_logpZ( GroupIDs, LP ) - self.E_logqZ( GroupIDs, LP ) \
           + self.E_logpW( LP )   - self.E_logqW(LP)

  ####################################################  ELBO computations
  def E_logpZ( self, GroupIDs, LP ):
    ElogpZ = 0
    for gg in xrange( len(GroupIDs) ):
      ElogpZ += np.sum( LP['resp'][GroupIDs[gg]] * LP['Elogw_perGroup'][gg,:-1] )
    return ElogpZ
    
  def E_logqZ( self, GroupIDs, LP ):
    ElogqZ = np.sum( LP['resp'] * np.log(EPS+LP['resp'] ) )
    return  ElogqZ    

  def E_logpW( self, LP ):
    nGroup = len(LP['alpha_perGroup'])
    ElogpW = gammaln( self.gamma ) - np.sum(gammaln(self.gamma*self.Ebeta))    
    ElogpW *= nGroup  # same prior over each group of data!
    ElogpW += np.dot( LP['Elogw_perGroup'], self.Ebeta-1 ).sum()
    return ElogpW
 
  def E_logqW( self, LP ):
    ElogqW = 0
    for gg in xrange( len(LP['alpha_perGroup']) ):
      a_gg = LP['alpha_perGroup'][gg]
      ElogqW +=  gammaln(  a_gg.sum()) - gammaln(  a_gg ).sum() \
                  + np.inner(  a_gg -1,  LP['Elogw_perGroup'][gg] )
    return ElogqW

  def E_logpV( self ):
    ''' E_q[ log Beta(v|1, a0) ] given point estimate q()=v*
    '''
    logZprior =gammaln(self.alpha0+self.alpha1)-gammaln(self.alpha0)-gammaln(self.alpha1 )
    logEterms  = (self.alpha1-1)*np.log(self.vstar) + (self.alpha0-1)*np.log(1-self.vstar)
    if self.truncType == 'z':
	    return self.K*logZprior + logEterms.sum()    
    elif self.truncType == 'v':
      return self.K*logZprior + logEterms[:-1].sum()

  def E_logqV( self ):
    ''' E_q[ log q(v) ] given point estimate q()=v*
    '''
    return 0
