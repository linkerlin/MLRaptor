'''
 Abstract class for learning algorithms for expfam models

  Simply defines some generic routines for
    ** initialization (see the init module)
    ** saving global parameters
    ** assessing convergence
    ** printing progress updates to stdout

Author: Mike Hughes (mike@michaelchughes.com)
'''
import numpy as np
import time
import os

from ..init.GaussObsSetInitializer import GaussObsSetInitializer

class LearnAlg(object):

  def __init__( self, savefilename='results/trace', nIter=100, \
                    initname='kmeans',  convTHR=1e-10, \
                    printEvery=5, saveEvery=5, evidenceEvery=5, \
                    doVerify=False, \
                    rhodelay=1, rhoexp=0.6, \
                    **kwargs ):
    self.savefilename = savefilename
    self.initname = initname
    self.convTHR = convTHR
    self.Niter = nIter
    self.printEvery = printEvery
    self.saveEvery = saveEvery
    self.evidenceEvery = evidenceEvery
    self.SavedIters = dict()
    self.doVerify = doVerify
    self.rhodelay =rhodelay
    self.rhoexp   = rhoexp

  def init_global_params( self, Data, seed ):
    obsType = self.expfamModel.obsModel.get_info_string()
    if obsType.count('Gauss') > 0:
      InitEngine = GaussObsSetInitializer( initname=self.initname, seed=seed)
      InitEngine.init_global_params( self.expfamModel, Data )
    elif obsType.count('Bern') > 0:
      InitEngine = GaussObsSetInitializer( initname=self.initname, seed=seed)
      InitEngine.init_global_params( self.expfamModel, Data )
      
  def fit( self, Data, seed):
    pass

  def save_state( self, iterid, evBound, doFinal=False):
    if iterid in self.SavedIters:
      return      
    mode = 'a'
    if doFinal or ( iterid % (self.saveEvery)==0 ):
      filename, ext = os.path.splitext( self.savefilename )
      self.SavedIters[iterid] = True
      with open( filename+'.iters', mode) as f:        
        f.write( '%d\n' % (iterid) )
      with open( filename+'.evidence', mode) as f:        
        f.write( '%.8e\n' % (evBound) )
      self.expfamModel.save_params( filename )

  ##################################################### Logging methods
  def calc_evidence( self, Data, SS, LP, Dtest=None):
    if Dtest is None:
      evBound = self.expfamModel.calc_evidence( Data, SS, LP )
    else:
      tLP = self.expfamModel.calc_local_params( Dtest )
      tSS = self.expfamModel.get_global_suff_stats( Dtest, tLP)
      evBound = self.expfamModel.calc_evidence( Dtest, tSS, tLP )
    return evBound
    
  def verify_evidence(self, evBound, prevBound):
    isValid = prevBound < evBound or np.allclose( prevBound, evBound, rtol=self.convTHR )
    if not isValid:
      print 'WARNING: evidence decreased!'
      print '    prev = % .15e' % (prevBound)
      print '     cur = % .15e' % (evBound)
    isConverged = np.abs(evBound-prevBound)/np.abs(evBound) <= self.convTHR
    return isConverged 

  def print_state( self, iterid, evBound, doFinal=False, status='', rho=None):
    doPrint = iterid % self.printEvery==0
    if rho is None:
      rhoStr = ''
    else:
      rhoStr = '%.4f |' % (rho)
    logmsg  = '  %5d/%s after %6.0f sec. | %s evidence % .9e' % (iterid, str(self.Niter), time.time()-self.start_time, rhoStr, evBound)
    if iterid ==0 and not doFinal:
      print '  Initialized via %s.' % (self.initname)
      print logmsg
    elif (doFinal and not doPrint):
      print logmsg
    elif (not doFinal and doPrint):
      print logmsg
    if doFinal:
      print '... done. %s' % (status)