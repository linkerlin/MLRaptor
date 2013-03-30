'''  This script will take two arguments:
      (1) file path location of a saved model 
      (2) filepath location of held-out data to process
               using all files in that directory,
'''
import numpy as np
import data.EasyToyGMMData as Toy
import argparse
import expfam
import glob
import os
from distutils.dir_util import mkpath  #mk_dir -p functionality

def DataGenerator( datapath ):
  for fname in glob.glob( os.path.join( datapath, '*.dat') ):
    print fname
    X = np.loadtxt( fname )
    GIDs = list( )
    GIDs.append( (0, X.shape[0]) )
    dummypath,shortname = os.path.split( fname )
    shortname = os.path.splitext( shortname)[0]
    yield shortname+'.wHat.txt', dict(X=X, GroupIDs=GIDs, nGroup=1)

def FakeDataGenerator( datapath ):
  for n in xrange( 5 ):
    Data = Toy.get_data_by_groups( nGroup=1, seed=n )  
    yield 'wHat%05d.txt'%(n), Data
    
def main( modelpath, datapath ):
  # Create inference engine from saved model
  expfammodel = expfam.ExpFamModel.BuildFromMatfile( modelpath )
  inferEngine = expfam.learn.VBInferHeldout( expfammodel, nIter=10, saveEvery=-1, printEvery=-1 )
  for dname, Data in DataGenerator( datapath ):
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