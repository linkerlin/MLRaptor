import scipy.io

def get_data(seed=867, **kwargs):
  try:
    Ddict = scipy.io.loadmat( '/Users/mhughes/git/MLRaptor/scripts/CompareMvsPy/BerkSegTrain25.mat')
  except IOError:
    Ddict = scipy.io.loadmat( '/home/mhughes/git/MLRaptor/scripts/CompareMvsPy/BerkSegTrain25.mat')
  Data = dict()
  Data['X'] = Ddict['X'].copy() # copy aligns things properly
  Data['nObs'] = Data['X'].shape[0]
  return Data

def print_data_info( modelName, **kwargs):
  print 'Berkeley Segmentation Patch Data'
  print '  First 25 images. All 8x8 patches.'

