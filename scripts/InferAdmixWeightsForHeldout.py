'''  This script will take two arguments:
      (1) file path location of a saved model 
      (2) filepath location of held-out data to process
               using all files in that directory,
'''

import argparse
import expfam

def DataGenerator( datapath ):
  pass

def main( modelpath, datapath ):
  # Create inference engine from saved model
  expfammodel = expfam.ExpFamModel.BuildFromMatfile( modelpath )
  inferEngine = expfam.learn.VBInferHeldout( expfammodel, nIter=10, saveEvery=-1, printEvery=-1 )
  for doutpath, Dchunk in DataGenerator( datapath ):
    LP = inferEngine.infer( Data )
    wHat = LP['alpha_perGroup']
    wHat /= wHat.sum()
    with open( doutpath, 'w') as f:
      f.write( expfam.util.MLUtil.np2flatstr(wHat)+'\n')    


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument( 'modelpath', type=str)
  parser.add_argument( 'datapath', type=str)
  args = parser.parse_args()
