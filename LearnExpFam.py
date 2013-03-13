#! /home/mhughes/mypy/epd64/bin/python
#$ -S /home/mhughes/mypy/epd64/bin/python
# ------ set working directory
#$ -cwd 
# ------ attach job number
#$ -j n
# ------ send to particular queue
#$ -o ../logs/$JOB_ID.$TASK_ID.out
#$ -e ../logs/$JOB_ID.$TASK_ID.err
'''
 User-facing executable script for learning Exp Family Models
  with a variety of possible inference algorithms, such as
    ** Expectation Maximization (EM)
    ** Variational Bayesian Inference (VB)

 Author: Mike Hughes (mike@michaelchughes.com)

  Quickstart
  -------
  To run EM for a 3-component GMM on easy toy data, do
  >> python LearnExpFam.py EasyToyGMMData MixModel Gaussian EM --K=3

  To run Variation Bayes on the same data using more components, do
  >> python LearnExpFam.py EasyToyGMMData MixModel Gaussian VB --K=10
 
  To run Variational Bayes on some simple binary toy data,
  >> python LearnExpFam.py EasyToyBernData MixModel Bernoulli VB --K=5
 
  Usage
  -------
  python LearnGMM.py <data_module_name> <aModel name> <eModel name> <alg name>  [options]

  <data_module> is a python script that lives in GMM/data/
     with either/both of the following functions
           * get_data()             for batch algorithms
           * minibatch_generator()  for online algorithms
     for example, see EasyToyGMMData.py

  <alg_name> is one of:
      EM : expectation maximization
      VB : variational bayes

  [options]  includes these and more
      --jobname : string name of the current experiment
      --nTask   : # separate initializations to try
      --nIter   : # iterations per task
      --K       : # mixture components to use 
      
      --saveEvery : # iters between saving global model params to disk
      --printEvery: # iters between printing progress update to stdout
'''
from distutils.dir_util import mkpath  #mk_dir functionality
import argparse
import os.path
import sys

import numpy as np

#############################################################
#             Code to Make Grid IO Possible
#############################################################
class MyLogFile(object):
  def __init__(self, fileobj):
    # reopen stdout file descriptor with write mode
    # and 0 as the buffer size (unbuffered)
    self.file = os.fdopen( fileobj.fileno(), 'w', 0)  

  def flush( self ):
    self.file.flush()

  def __getattr__(self, attr):
    return getattr( self.file, attr )

  def write( self, data):
    self.file.write( data )
    self.file.flush()
    os.fsync( self.file.fileno() )
 
  def fileno( self ):
    return self.file.fileno()

  def close( self ):
    self.file.close()

def clear_folder( savefolder, prefix=None ):
  #print 'Clearing %s' % (savefolder)
  for the_file in os.listdir( savefolder ):
    if prefix is not None:
      if not the_file.startswith(prefix):
        continue
    file_path = os.path.join( savefolder, the_file)
    if os.path.isfile(file_path) or os.path.islink(file_path):
      os.unlink(file_path)


jobID = 1
taskID = 1

if not sys.stdout.isatty():
  sys.stdout = MyLogFile( sys.stdout )
  sys.stderr = MyLogFile( sys.stderr )
  os.chdir('..')
  sys.path[0] = os.getcwd()
  print 'This is LearnExpFam.py'
  print 'Python version %d.%d.%d' % sys.version_info[ :3]
  print 'Numpy version %s' % (np.__version__)
  print 'Cur Dir:', os.getcwd()
  print 'Local search path:', sys.path[0]
  try:
    jobID = int(  os.getenv( 'JOB_ID' ) )
    taskID = int( os.getenv( 'SGE_TASK_ID' ) )
    LOGFILEPREFIX = os.path.join( os.getcwd(), 'logs', str(jobID)+'.'+str(taskID) )
  except TypeError:
    pass
  print 'JobID  %d' % (jobID )
  print 'TaskID %d' % (taskID )
  print '---------------------------------------------'

#############################################################
#             Code to Parse Arguments
#############################################################

import expfam as ef
AllocModelConstructor = {'MixModel': ef.mix.MixModel, \
                         'DPMixModel': ef.mix.DPMixModel, \
                         'AdmixModel': ef.admix.AdmixModel, \
                         'Admix': ef.admix.AdmixModel, \
                         'HDPAdmixModel': ef.admix.HDPAdmixModel,\
                         'HDP': ef.admix.HDPAdmixModel}

PriorConstr = {'Gaussian': ef.obsModel.GaussWishDistrIndep, \
               'Gauss': ef.obsModel.GaussWishDistrIndep, \
               'Normal': ef.obsModel.GaussWishDistrIndep, \
               'Bern': ef.obsModel.BetaDistr, \
               'Bernoulli': ef.obsModel.BetaDistr}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( 'datagenModule', type=str )
    parser.add_argument( 'modelName', type=str )
    parser.add_argument( 'obsName', type=str )
    parser.add_argument( 'algName', type=str )

    parser.add_argument( '--K', type=int, default=3 )
    parser.add_argument( '--alpha0', type=float, default=1.0 )
    parser.add_argument( '--min_covar', type=float, default=1e-9 )

    parser.add_argument( '--doprior', action='store_true', default=False )

    lgroup = parser.add_mutually_exclusive_group()
    lgroup.add_argument('--dobatch', action='store_true',default=True)
    lgroup.add_argument('--doonline', action='store_true')

    parser.add_argument('--dotest', action='store_true',default=False)

    # Batch learning args
    parser.add_argument( '--nIter', type=int, default=100 )

    # Online learning args
    parser.add_argument( '--batch_size', type=int, default=100 )
    parser.add_argument( '--nBatch', type=int, default=50 )
    parser.add_argument( '--nRep', type=int, default=1 )    
    parser.add_argument( '--rhoexp', type=float, default=0.5 )
    parser.add_argument( '--rhodelay', type=float, default=1 )

    # Generic args
    parser.add_argument( '--jobname', type=str, default='defaultjob' )
    parser.add_argument( '--taskid', type=int, default=taskID )
    parser.add_argument( '--nTask', type=int, default=1 )

    parser.add_argument( '--initname', type=str, default='random' )
    parser.add_argument( '--seed', type=int, default=8675309 )    
    parser.add_argument( '--evidenceEvery', type=int, default=1 )
    parser.add_argument( '--printEvery', type=int, default=5 )
    parser.add_argument( '--saveEvery', type=int, default=10 )
    return parser.parse_args()

def main(args):        
    ####################################################### Data Module parsing
    dataParams = dict()
    for argName in ['nBatch', 'nRep', 'batch_size', 'seed']:
      dataParams[argName] = args.__getattribute__( argName )
    # Dynamically load module provided by user as data-generator
    datagenmod = __import__( 'data.' + args.datagenModule, fromlist=['data'])
    if 'print_data_info' in dir( datagenmod ):
      datagenmod.print_data_info()
      
    ####################################################### Algorithm settings
    algParams = dict()
    for argName in ['initname', 'nIter', 'rhoexp', 'rhodelay', \
                    'nIter', 'printEvery', 'saveEvery','evidenceEvery']:
      algParams[ argName ] = args.__getattribute__( argName ) 
      
    if args.doonline:
      algName = 'o'+args.algName
    else:
      algName = args.algName
    print 'Learn Alg:  %s' % (algName)
    
    ####################################################### ExpFam Model Params
    modelParams = dict()
    for argName in ['K', 'alpha0', 'min_covar']:
      modelParams[ argName ] = args.__getattribute__( argName ) 

    obsPriorParams = dict()
    for argName in []:
      obsPriorParams[ argName ] = args.__getattribute__( argName ) 
    if args.doprior or args.algName.count('VB')>0:
      obsPrior = PriorConstr[ args.obsName ]( **obsPriorParams )
    else:
      obsPrior = None  
    am = AllocModelConstructor[ args.modelName ]( qType=algName, **modelParams )
    m = ef.ExpFamModel( am, args.obsName, obsPrior )
    m.print_model_info()

    doAdmix = (args.modelName.count('Admix') + args.modelName.count('HDP') )> 0

    jobpath = os.path.join(args.datagenModule[:7], args.modelName, algName, args.jobname)
    
    ####################################################### Spawn individual tasks
    for task in xrange( args.taskid, args.taskid+args.nTask ):    
      seed = hash( args.jobname+str(task) ) % np.iinfo(int).max
      algParams['seed'] = seed

      basepath = os.path.join( 'results', jobpath, str(task) )
      mkpath(  basepath )
      clear_folder( basepath )
      algParams['savefilename'] = os.path.join( basepath, 'trace' )

      print 'Trial %2d/%d | alg. seed: %d | data seed: %d' \
                 % (task, args.nTask, algParams['seed'], dataParams['seed'])
      print '  savefile: %s' % (algParams['savefilename'])

      if jobID > 1:
        logpath = os.path.join( 'logs', jobpath )
        mkpath( logpath )
        clear_folder( logpath, prefix=str(task) )
        os.symlink( LOGFILEPREFIX+'.out', '%s/%d.out' % (logpath, task) )
        os.symlink( LOGFILEPREFIX+'.err', '%s/%d.err' % (logpath, task) )
        print '   logfile: %s' % (logpath)
    

      ##########################################################  Build Learning Alg
      '''if args.algName.count('EM')>0:
        learnAlg = ef.learn.EMLearnAlg( m, **algParams )

      elif args.algName.count('VB')>0:
        if args.doonline:
          learnAlg = ef.learn.OnlineVBLearnAlg( m, **algParams )
        elif args.dobatch:
          learnAlg = ef.learn.VBLearnAlg( m, **algParams )
      '''
      if args.doonline:
        learnAlg = ef.learn.OnlineVBLearnAlg( m, **algParams )
      elif args.dobatch:
        learnAlg = ef.learn.VBLearnAlg( m, **algParams )
      
      if args.doonline:
        if doAdmix:
          Data = datagenmod.group_minibatch_generator( **dataParams )
        else:
          Data = datagenmod.minibatch_generator( **dataParams )
      else:
        if doAdmix:
          Data = datagenmod.get_data_by_groups( **dataParams )
        else:
          Data = datagenmod.get_data( **dataParams )

      if args.dotest:
        testParams = dataParams
        testParams['seed'] += 1
        if doAdmix:
          Dtest = datagenmod.get_data_by_groups( **testParams )
        else:
          Dtest = datagenmod.get_data( **testParams )
        learnAlg.fit( Data, seed, Dtest=Dtest)
      else:
        learnAlg.fit( Data, seed )

if __name__ == '__main__':
  args = parse_args()
  main(args)
