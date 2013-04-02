import argparse
import glob
import os
import numpy as np
import ImgPatchHandler
import FloReader
import itertools

POpts = dict( border=0, Lx=10, Ly=10, dLx=10, dLy=10)

def extract_patches_from_dataset( flodir, outdir, ext='flo', doZip=False, doSubtractDC=True):
  if doZip:
    outext = '.dat.gz'
  else:
    outext = '.dat'

  fileList = sorted( glob.glob( os.path.join( flodir, '*.' +ext) ) )
  for fpath in fileList:
    basename = os.path.split( fpath )[-1]
    basename = os.path.splitext(basename)[0]
    savefname = os.path.join( outdir, basename+outext )
    print basename
    U,V = FloReader.readFlowFile( fpath )

    hU = ImgPatchHandler.ImgPatchHandler( U, POpts )
    UpatchList = list()
    for pvec in hU.FlatPatchGenerator():
      if doSubtractDC:
        pvec = pvec - pvec.mean()
      UpatchList.append(  pvec )

    hV = ImgPatchHandler.ImgPatchHandler( V, POpts )
    VpatchList = list()
    for pvec in hV.FlatPatchGenerator():
      if doSubtractDC:
        pvec = pvec - pvec.mean()
      VpatchList.append(  pvec )
    
    patchList = [np.hstack(pair) for pair in itertools.izip(UpatchList,VpatchList)]
    
    np.savetxt( savefname,  np.vstack( patchList ), fmt='% 14.8f' )
    
def main():
  Parser = argparse.ArgumentParser()
  Parser.add_argument( 'flodir', type=str, default='.' )
  Parser.add_argument( 'outdir', type=str, default='.' )
  args = Parser.parse_args()
  
  extract_patches_from_dataset(  args.flodir, args.outdir)

if __name__ == '__main__':
  main()
