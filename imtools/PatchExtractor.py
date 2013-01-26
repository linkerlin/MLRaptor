import argparse
import glob
import os
import numpy as np
import ImgPatchHandler

def np2flatstr( X, fmt='% 15.8f' ):
  return '\t'.join( [fmt % x for x in X.flatten() ] )  

def extract_patches_from_dataset( imgdir, outdir, ext='jpg', doZip=False ):
  if doZip:
    outext = '.dat.gz'
  else:
    outext = '.dat'

  fileList = glob.glob( os.path.join(imgdir, '*.' +ext) )
  for fpath in fileList:
    h = ImgPatchHandler.ImgPatchHandler( fpath )

    basename = os.path.split( fpath )[-1]
    basename = os.path.splitext(basename)[0]
    savefname = os.path.join( outdir, basename+outext )

    patchList = list()
    for pvec in h.FlatPatchGenerator():
      pvec = pvec - pvec.mean()
      patchList.append(  pvec )
    np.savetxt( savefname,  np.vstack( patchList ) )
    '''
    with open( outdir+basename+".dat", 'w') as f:
      for pvec in h.FlatPatchGenerator():
        pvec = pvec - pvec.mean()
        f.write( np2flatstr(pvec) +'\n' )
    '''
    
def main():
  Parser = argparse.ArgumentParser()
  Parser.add_argument( 'imgdir', type=str, default='.' )
  Parser.add_argument( 'outdir', type=str, default='.' )
  args = Parser.parse_args()
  
  extract_patches_from_dataset(  args.imgdir, args.outdir)

if __name__ == '__main__':
  main()
