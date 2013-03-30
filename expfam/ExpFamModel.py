'''
  Represents standard hierarchical Bayesian model
    with conditional distributions in Exponential Family

 Author: Mike Hughes (mike@michaelchughes.com)

 Parameters
 -------
   allocModel : name or object
   obsType   :  name or object
   obsPrior   : object [default None]
     
 Inference
 -------
  See VBLearnAlg.py
'''

import numpy as np
import scipy.io
import os

from .obsModel import *
from .mix import *
from .admix import *

class ExpFamModel( object ):

  def __init__( self, allocModel, obsModelName, obsPrior=None, **kwargs ):
    self.allocModel = allocModel
    self.qType = self.allocModel.qType
    self.K     = self.allocModel.K
    if type( obsModelName ) == str:
      if obsModelName == 'Gaussian':
        self.obsModel   = GaussObsCompSet( allocModel.K, allocModel.qType, obsPrior)
      if obsModelName == 'Bernoulli':
        self.obsModel   = BernObsCompSet( allocModel.K, allocModel.qType, obsPrior)
      if obsModelName.count( 'Mult' ) >0:
        self.obsModel   = MultObsCompSet( allocModel.K, allocModel.qType, obsPrior)
    else:
      self.obsModel = obsModelName
  
  #===================================================
  def set_obs_dims( self, Data):
    self.obsModel.set_obs_dims( Data )
  
  def calc_evidence( self, Data, SS, LP):
    return self.allocModel.calc_evidence( Data, SS, LP) \
          + self.obsModel.calc_evidence( Data, SS, LP)
   
  def calc_local_params( self, Data, LP=None):
    if LP is None:
      LP = dict()
    LP = self.obsModel.calc_local_params( Data, LP ) #collect log soft evidence
    LP = self.allocModel.calc_local_params( Data, LP )
    return LP
  
  def get_global_suff_stats( self, Data, LP):
    SS = dict()
    SS = self.allocModel.get_global_suff_stats( Data, SS, LP )
    SS = self.obsModel.get_global_suff_stats( Data, SS, LP )
    return SS
    
  def update_global_params( self, SS, rho=None, **kwargs):
    self.allocModel.update_global_params(SS, rho, **kwargs)
    self.obsModel.update_global_params( SS, rho, **kwargs)

  #===================================================  PRINT TO STDOUT
  def print_model_info( self ):
    print 'Allocation Model:  %s'%  (self.allocModel.get_info_string() )
    print 'Obs. Data  Model:  %s'%  (self.obsModel.get_info_string() )
    print 'Obs. Data  Prior:  %s'%  (self.obsModel.get_info_string_prior() )
  
  def print_global_params( self ):
    print 'Allocation Model:'
    print  self.allocModel.get_human_global_param_string()
    print 'Obs. Data Model:'
    print  self.obsModel.get_human_global_param_string()

  #===================================================  SAVE TO FILE
  def save_params(self, fname, saveext='mat', doSavePriorInfo=False):
    self.save_alloc_params( fname, saveext)
    self.obsModel.save_params(fname, saveext)
    if doSavePriorInfo:
      # Remove "Iter000xx" prefix from fname!
      fnameList = fname.split( os.path.sep )[:-1]
      fname = os.path.sep.join( fnameList )
      self.save_alloc_prior( fname, saveext)
      self.save_obs_prior( fname, saveext)

  def save_alloc_params( self, fname, saveext):
    if saveext == 'txt':
      outpath = fname + 'AllocModel.txt'
      astr = self.allocModel.to_string()
      if len(astr) == 0:
        return None
      with open( outpath, 'a') as f:
        f.write( astr + '\n')
    elif saveext == 'mat':
      outpath = fname + 'AllocModel.mat'
      adict = self.allocModel.to_dict()
      if len( adict.keys() ) == 0:
        return None
      scipy.io.savemat( outpath, adict, oned_as='row')

  def save_alloc_prior( self, fname, saveext):
    if saveext == 'txt':
      outpath = os.path.join( fname, 'AllocPrior.txt')
      astr = self.allocModel.to_prior_string()
      if len(astr) == 0:
        return None
      with open( outpath, 'a') as f:
        f.write( astr + '\n')
    elif saveext == 'mat':      
      outpath = os.path.join( fname, 'AllocPrior.mat')
      adict = self.allocModel.get_prior_dict()
      if len( adict.keys() ) == 0:
        return None
      scipy.io.savemat( outpath, adict, oned_as='row')

  def save_obs_prior( self, fname, saveext):
    if saveext == 'txt':
      outpath = os.path.join( fname, 'ObsPrior.txt')
      astr = self.obsModel.to_prior_string()
      if len(astr) == 0:
        return None
      with open( outpath, 'a') as f:
        f.write( astr + '\n')
    elif saveext == 'mat':      
      outpath = os.path.join( fname, 'ObsPrior.mat')
      adict = self.obsModel.get_prior_dict()
      if len( adict.keys() ) == 0:
        return None
      scipy.io.savemat( outpath, adict, oned_as='row')

  @classmethod
  def BuildFromMatfile( self, matfilepath):
    # NEED TO STORE MODELTYPES for Alloc and Obs in the matfilepath!
