'''
 User-facing executable script for learning Gaussian Mixture Models
  with a variety of possible inference algorithms, such as
    ** Expectation Maximization (EM)
    ** Variational Bayesian Inference (VB)

 Author: Mike Hughes (mike@michaelchughes.com)

  Quickstart
  -------
  To run EM for a 3-component GMM on easy toy data, do
  >> python LearnGMM.py EasyToyGMMData EM --K=3

  To run Variation Bayes on the same data, do
  >> python LearnGMM.py EasyToyGMMData VB --K=3
 
  Usage
  -------
  python LearnGMM.py <data_module> <alg_name>  [options]

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
import sklearn.mixture

import GMM.EMLearnerGMM as EM
import GMM.OnlineEMLearnerGMM as OEM
import GMM.VBLearnerGMM as VB
import GMM.GMM as GMM
import GMM.QGMM as QGMM
import GMM.QDPGMM as QDPGMM
import GMM.GaussWishDistr as GaussWishDistr

# reopen stdout file descriptor with write mode
# and 0 as the buffer size (unbuffered)
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument( 'datagenModule', type=str )
    Parser.add_argument( 'algName', type=str )
    #TODO:Parser.add_argument( 'modelName', type=str )

    Parser.add_argument( '-K', '--K', type=int, default=3 )

    Parser.add_argument( '--alpha0', type=float, default=1.0 )
    Parser.add_argument( '--covar_type', type=str, default='full' )
    Parser.add_argument( '--min_covar', type=float, default=1e-9 )

    # Batch learning args
    Parser.add_argument( '--nIter', type=int, default=100 )

    # Online learning args
    Parser.add_argument( '--batch_size', type=int, default=100 )
    Parser.add_argument( '--nBatch', type=int, default=50 )
    Parser.add_argument( '--nRep', type=int, default=1 )    
    Parser.add_argument( '--rhoexp', type=float, default=0.5 )
    Parser.add_argument( '--rhodelay', type=float, default=1 )

    # Generic args
    Parser.add_argument( '--jobname', type=str, default='defaultjob' )
    Parser.add_argument( '--taskid', type=int, default=1 )
    Parser.add_argument( '--nTask', type=int, default=1 )

    Parser.add_argument( '--initname', type=str, default='random' )
    Parser.add_argument( '--seed', type=int, default=8675309 )    
    Parser.add_argument( '-v', '--doVerbose', action='store_true', default=False )
    Parser.add_argument( '--printEvery', type=int, default=5 )
    Parser.add_argument( '--saveEvery', type=int, default=10 )
    Parser.add_argument( '--doProfile', action='store_true', default=False )
    args = Parser.parse_args()

    modelParams = dict()
    for argName in ['K', 'covar_type', 'min_covar', 'alpha0']:
      modelParams[ argName ] = args.__getattribute__( argName ) 

    dataParams = dict()
    for argName in ['nBatch', 'nRep', 'batch_size', 'seed']:
      dataParams[argName] = args.__getattribute__( argName )

    algParams = dict()
    for argName in ['initname', 'nIter', 'rhoexp', 'rhodelay', \
                    'nIter', 'printEvery', 'saveEvery']:
      algParams[ argName ] = args.__getattribute__( argName ) 

    # Dynamically load module provided by user as data-generator
    #   this must implement a generator function called "minibatch_generator" or "get_data"
    datagenmod = __import__( 'GMM.data.' + args.datagenModule, fromlist=['GMM','data'])
    if 'print_data_info' in dir( datagenmod ):
      datagenmod.print_data_info()

    gmm = GMM.GMM( **modelParams )
    gmm.print_model_info()


    for task in xrange( args.taskid, args.taskid+args.nTask ):    
      basepath = os.path.join( 'results', args.algName, args.jobname, str(task) )
      mkpath(  basepath )
      algParams['savefilename'] = os.path.join( basepath, 'trace' )
      seed = hash( args.jobname+str(task) ) % np.iinfo(int).max
      algParams['seed'] = seed
    
      print 'Trial %2d/%d | savefile: %s | seed: %d' % (task, args.nTask, algParams['savefilename'], algParams['seed'])

      if args.algName.startswith( 'o' ) and  args.algName.count('EM')>0:
        DataGen = datagenmod.minibatch_generator( **dataParams )    
        gmm = GMM.GMM( **modelParams )
        em = OEM.OnlineEMLearnerGMM( gmm, **algParams )
        em.fit(  DataGen, seed )
      elif args.algName.count('sklearnEM') > 0:
        sklgmm = sklearn.mixture.GMM(  n_components=args.K, random_state=seed, covariance_type=args.covar_type, \
                                       min_covar=args.min_covar, n_init=1, n_iter=args.nIter, init_params='' )
        X = datagenmod.get_data( **dataParams )

        gmm = GMM.GMM( **modelParams )
        em = EM.EMLearnerGMM( gmm, **algParams )
        em.init_params( X, seed=seed )
        
        sklgmm.weights_ = gmm.w
        sklgmm.means_   = gmm.mu
        sklgmm.covars_   = gmm.Sigma

        sklgmm.fit( X )
      elif args.algName.count('EM') > 0:
        Data    = datagenmod.get_data( **dataParams )    
        gmm = GMM.GMM( **modelParams )
        em = EM.EMLearnerGMM( gmm, **algParams )
        em.fit( Data, seed )
      elif args.algName.count('DPVB') > 0:
        Data    = datagenmod.get_data( **dataParams )    
        D = Data.shape[1]
        gw   = GaussWishDistr.GaussWishDistr( D=D )
        qdp = QDPGMM.QDPGMM( gw, **modelParams )
        em = VB.VBLearnerGMM( qdp, **algParams )
        em.fit(  Data, seed )  
      elif args.algName.count('VB') > 0:
        Data    = datagenmod.get_data( **dataParams )    
        D = Data.shape[1]
        dF = D+1
        invW = np.eye(D)
        gw   = GaussWishDistr.GaussWishDistr( D=D )
        qgmm = QGMM.QGMM( gw, **modelParams )
        em = VB.VBLearnerGMM( qgmm, **algParams )
        em.fit(  Data, seed )  


if __name__ == '__main__':
  main()
