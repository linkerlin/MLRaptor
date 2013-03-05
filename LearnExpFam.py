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

  To run Variation Bayes on the same data, do
  >> python LearnExpFam.py EasyToyGMMData MixModel Gaussian EM --K=3
 
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

  [options]  includes
      --jobname : string name of the current experiment
      --nTask   : # separate initializations to try
      --nIter   : # iterations per task
      --K       : # mixture components to use 
'''
import argparse

import os.path
import sys
from distutils.dir_util import mkpath  #mk_dir p functionality

import numpy as np
#import sklearn.mixture

import ExpFam as ef

# reopen stdout file descriptor with write mode
# and 0 as the buffer size (unbuffered)
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

def main():
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
    parser.add_argument( '--taskid', type=int, default=1 )
    parser.add_argument( '--nTask', type=int, default=1 )

    parser.add_argument( '--initname', type=str, default='random' )
    parser.add_argument( '--seed', type=int, default=8675309 )    
    parser.add_argument( '--printEvery', type=int, default=5 )
    parser.add_argument( '--saveEvery', type=int, default=10 )
    args = parser.parse_args()
    
    modelParams = dict()
    for argName in ['K', 'alpha0', 'min_covar']:
      modelParams[ argName ] = args.__getattribute__( argName ) 

    obsPriorParams = dict()
    for argName in []:
      obsPriorParams[ argName ] = args.__getattribute__( argName ) 

    dataParams = dict()
    for argName in ['nBatch', 'nRep', 'batch_size', 'seed']:
      dataParams[argName] = args.__getattribute__( argName )

    algParams = dict()
    for argName in ['initname', 'nIter', 'rhoexp', 'rhodelay', \
                    'nIter', 'printEvery', 'saveEvery']:
      algParams[ argName ] = args.__getattribute__( argName ) 

    # Dynamically load module provided by user as data-generator
    datagenmod = __import__( 'data.' + args.datagenModule, fromlist=['data'])
    if 'print_data_info' in dir( datagenmod ):
      datagenmod.print_data_info()

    AllocModelConstructor = {'MixModel': ef.mix.MixModel, 'QMixModel': ef.mix.QMixModel, 'QAdmixModel': ef.admix.QAdmixModel, 'QHDPAdmix': ef.admix.QHDPAdmixModel}
    am = AllocModelConstructor[ args.modelName ]( **modelParams )

    PriorConstr = {'Gaussian': ef.obsModel.GaussWishDistr}
    if args.doprior or args.algName.count('VB')>0:
      obsPrior = PriorConstr[ args.obsName ]( **obsPriorParams )
    else:
      obsPrior = None
      
    m = ef.ExpFamModel( am, args.obsName, obsPrior )
    m.print_model_info()

    for task in xrange( args.taskid, args.taskid+args.nTask ):    
      basepath = os.path.join( 'results', args.algName, args.jobname, str(task) )
      mkpath(  basepath )
      algParams['savefilename'] = os.path.join( basepath, 'trace' )
      seed = hash( args.jobname+str(task) ) % np.iinfo(int).max
      algParams['seed'] = seed
    
      print 'Trial %2d/%d | savefile: %s | seed: %d' % (task, args.nTask, algParams['savefilename'], algParams['seed'])

      if args.algName.count('EM')>0:
        if args.doonline:
          learnAlg = ef.learn.OnlineEMLearnAlg( m, **algParams )
        elif args.dobatch:
          learnAlg = ef.learn.EMLearnAlg( m, **algParams )

      elif args.algName.count('VB')>0:
        if args.doonline:
          learnAlg = ef.learn.OnlineVBLearnAlg( m, **algParams )
        
        elif args.dobatch:
          learnAlg = ef.learn.VBLearnAlg( m, **algParams )
        
      if args.doonline:
        DataGen = datagenmod.minibatch_generator( **dataParams )
        learnAlg.fit( DataGen, seed) 
      else:
        if args.modelName.count('Admix') > 0:
          Data = datagenmod.get_data_by_groups( **dataParams )
        else:
          Data = datagenmod.get_data( **dataParams )
        learnAlg.fit( Data, seed)


if __name__ == '__main__':
  main()
