'''
  MixModel.py
     Bayesian parametric admixture model with a finite number of components K

  Provides code for performing variational Bayesian inference,
     using a mean-field approximation.
     
 Author: Mike Hughes (mike@michaelchughes.com)

 Parameters
 -------
    K        : # of components
    alpha0   : scalar hyperparameter of symmetric Dirichlet prior on mix. weights

 References
 -------
   Latent Dirichlet Allocation, by Blei, Ng, and Jordan
      introduces a classic admixture model with Dirichlet-Mult observations.
'''
import itertools
import numpy as np

import scipy.io
import os

from scipy.special import gammaln, digamma
from ..util.MLUtil import logsumexp, np2flatstr

EPS = 10*np.finfo(float).eps

class AdmixModel( object ):

  def __init__( self, K=3, alpha0=1.0, qType='VB', **kwargs):
    if qType.count('EM')>0:
      raise ValueError('AdmixModel cannot do EM. Only VB learning possible.')
    self.qType = qType
    self.K = K
    self.alpha0 = alpha0

  def get_info_string( self):
    return 'Finite admixture model with %d components | alpha=%.2f' % (self.K, self.alpha0)

  def get_human_global_param_string(self):
    ''' No global parameters! So just return blank line
    '''
    mystr = ''
    for rowID in xrange(3):
      mystr += np2flatstr( np.exp(self.Elogw[rowID]), '%3.2f') + '\n'
    return mystr

  def to_string( self):
    ''' No global parameters! So just return blank line
    '''
    return ''  
    	    	
  def to_dict( self ):
    return dict()  	    	
    	    	
  def get_prior_dict( self ):
    return dict( alpha0=self.alpha0, K=self.K )

  ###################################################################  Local Params (Estep)
  def calc_local_params( self, Data, LP):
    ''' E-step
          alternate between these updates until convergence
             q(Z)  (posterior on topic-token assignment)
         and q(W)  (posterior on group-topic distribution)
    '''
    try:
      LP['N_perGroup']
    except KeyError:
      LP['N_perGroup'] = np.zeros( (Data['nGroup'],self.K) )

    GroupIDs = Data['GroupIDs']
    nGroups = Data['nGroup']
    prevVec = None
    for rep in xrange( 10 ):
      LP = self.get_local_W_params( Data, LP)
      LP = self.get_local_Z_params( Data, LP)
      for gg in range( nGroups ):
        groupResp = LP['resp'][ GroupIDs[gg][0]:GroupIDs[gg][1] ]
        LP['N_perGroup'][gg] = np.sum( groupResp, axis=0 )

      curVec = LP['alpha_perGroup'].flatten()
      if prevVec is not None and np.allclose( prevVec, curVec ):
        break
      prevVec = curVec
    return LP
    
  def get_local_W_params( self, Data, LP):
    GroupIDs = Data['GroupIDs']
    alpha_perGroup = self.alpha0 + LP['N_perGroup']
    LP['alpha_perGroup'] = alpha_perGroup
    LP['Elogw_perGroup'] = digamma( alpha_perGroup ) \
                             - digamma( alpha_perGroup.sum(axis=1) )[:,np.newaxis]
    # Added this line to aid human inspection. self.Elogw is never used except to print status
    self.Elogw = LP['Elogw_perGroup']
    return LP
    
  def get_local_Z_params( self, Data, LP):
    GroupIDs = Data['GroupIDs']
    lpr = LP['E_log_soft_ev'].copy() # so we can do += later
    for gg in xrange( len(GroupIDs) ):
      lpr[ GroupIDs[gg][0]:GroupIDs[gg][1] ] += LP['Elogw_perGroup'][gg]
    lprPerItem = logsumexp( lpr, axis=1 )
    resp   = np.exp( lpr-lprPerItem[:,np.newaxis] )
    resp   /= resp.sum( axis=1)[:,np.newaxis] # row normalize
    assert np.allclose( resp.sum(axis=1), 1)
    if 'wordIDs_perGroup' in Data:
      for gg in xrange(len(GroupIDs)):
        resp[ GroupIDs[gg][0]:GroupIDs[gg][1] ] *= Data['wordCounts_perGroup'][gg][:,np.newaxis]
    LP['resp'] = resp
    return LP

  ##############################################################  Global Sufficient Stats
  def get_global_suff_stats( self, Data, SS, LP ):
    ''' Just count expected # assigned to each cluster across all groups, as usual
    '''
    SS['N'] = np.sum( LP['resp'], axis=0 )
    SS['Ntotal'] = SS['N'].sum()
    return SS
    
  ##############################################################  Global Param Updates    
  def update_global_params( self, SS, rho=None, **kwargs ):
    '''Admixtures have no global allocation params! 
         Mixture weights are group/document specific.
    '''
    pass

  ##############################################################  Evidence/ELBO calc.
  def calc_evidence( self, Data, SS, LP ):
    GroupIDs = Data['GroupIDs']
    if 'wordCounts_perGroup' in Data:
      respNorm = LP['resp'] / LP['resp'].sum(axis=1)[:,np.newaxis]
    else:
      respNorm = None
    return self.E_logpZ( GroupIDs, LP ) - self.E_logqZ( GroupIDs, LP, respNorm ) \
                 + self.E_logpW( LP )   - self.E_logqW(LP)

  def E_logpZ( self, GroupIDs, LP ):
    ElogpZ = 0
    for gg in xrange( len(GroupIDs) ):
      ElogpZ += np.sum( LP['resp'][ GroupIDs[gg][0]:GroupIDs[gg][1] ] * LP['Elogw_perGroup'][gg] )
    return ElogpZ
    
  def E_logqZ( self, GroupIDs, LP, respNorm=None ):
    if respNorm is None:
      ElogqZ = np.sum( LP['resp'] * np.log(EPS+LP['resp'] ) )
    else:
      ElogqZ = np.sum( LP['resp'] * np.log(EPS+respNorm ) )
    return  ElogqZ    


  def E_logpW( self, LP ):
    nGroup = len(LP['alpha_perGroup'])
    ElogpW = gammaln(self.K*self.alpha0)-self.K*gammaln(self.alpha0)    
    ElogpW *= nGroup  # same prior over each group of data!
    for gg in xrange( nGroup ):
      ElogpW += (self.alpha0-1)*LP['Elogw_perGroup'][gg].sum()
    return ElogpW
 
  def E_logqW( self, LP ):
    ElogqW = 0
    for gg in xrange( len(LP['alpha_perGroup']) ):
      a_gg = LP['alpha_perGroup'][gg]
      ElogqW +=  gammaln(  a_gg.sum()) - gammaln(  a_gg ).sum() \
                  + np.inner(  a_gg -1,  LP['Elogw_perGroup'][gg] )
    return ElogqW

  ######################################################  Factory Method: from mat file
  @classmethod
  def BuildFromMatfile( self, matfilepath, priormatfilepath=None ):
    if priormatfilepath is None:
      if matfilepath.endswith('.mat'):
        priormatfilepath = os.path.split( matfilepath )[0]
        priormatfilepath = os.path.join( priormatfilepath, 'AllocPrior.mat')
      else:
        priormatfilepath = os.path.join( matfilepath, 'AllocPrior.mat')
        matfilepath = os.path.join( matfilepath, 'BestAllocModel.mat')

    PriorDict = scipy.io.loadmat( priormatfilepath )
    alpha0 = float( PriorDict['alpha0'][0] )
    K = PriorDict['K'][0]
    return AdmixModel( K=K, alpha0=alpha0 )
