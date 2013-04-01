#! /home/mhughes/mypy/epd64/bin/python
#$ -S /home/mhughes/mypy/epd64/bin/python
# ------ set working directory
#$ -cwd 
# ------ attach job number
#$ -j n
# ------ send to particular queue
#$ -o ../logs/$JOB_ID.$TASK_ID.out
#$ -e ../logs/$JOB_ID.$TASK_ID.err

'''  This script will take two arguments:
      (1) file path location of a saved model 
      (2) filepath location of held-out data to process
               using all files in that directory,
'''

import sys
sys.path.append('/home/mhughes/git/MLRaptor/')

import numpy as np
import data.EasyToyGMMData as Toy
import argparse
import expfam
import glob
import os
from distutils.dir_util import mkpath  #mk_dir -p functionality

def DataGenerator( datapath, modelpath ):
  for fname in glob.glob( os.path.join( datapath, '*.dat') ):
    print fname
    dummypath,shortname = os.path.split( fname )
    shortname = os.path.splitext( shortname)[0]
    dname = shortname+'.wHat.txt'
    if os.path.exists( os.path.join( modelpath, 'heldout', dname) ):
      print '.... skipping'
      continue
    X = np.loadtxt( fname )
    GIDs = list( )
    GIDs.append( (0, X.shape[0]) )
    yield dname, dict(X=X, GroupIDs=GIDs, nGroup=1)

def FakeDataGenerator( datapath ):
  for n in xrange( 5 ):
    Data = Toy.get_data_by_groups( nGroup=1, seed=n )  
    yield 'wHat%05d.txt'%(n), Data
    
def main( modelpath, datapath ):
  # Create inference engine from saved model
  expfammodel = expfam.ExpFamModel.BuildFromMatfile( modelpath )
  inferEngine = expfam.learn.VBInferHeldout( expfammodel, nIter=10, saveEvery=-1, printEvery=-1 )
  for dname, Data in DataGenerator( datapath, modelpath ):
    LP = inferEngine.infer( Data )
    wHat = LP['alpha_perGroup']
    wHat /= wHat.sum()
    doutpath = os.path.join( modelpath, 'heldout')
    mkpath( doutpath )
    doutpath = os.path.join( doutpath, dname)
    print dname
    with open( doutpath, 'w') as f:
      f.write( expfam.util.MLUtil.np2flatstr(wHat)+'\n')    


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument( 'modelpath', type=str)
  parser.add_argument( 'datapath', type=str)
  args = parser.parse_args()
  main( args.modelpath, args.datapath)
