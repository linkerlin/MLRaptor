'''
Module for reading .flo files
'''
import numpy as np
import Image

def readFlowFile( filename ):
  with open( filename, 'rb') as f:
    header = np.fromfile( f, dtype=np.float32, count=1)
    if header[0] != 202021.25:
      raise ValueError('Invalid .flo file %s' % (filename) )
    dims = np.fromfile( f, dtype=np.uint32, count=2)
    W,H = dims
    if W < 1 or W > 99999 or H < 1 or H > 99999:
      raise ValueError('Invalid dimensions: H=%d, W=%d' % (H,W) )
    flowData = np.fromfile( f, dtype=np.float32)
  flowData = flowData.reshape( (H, W, 2) )

  FlowX = flowData[:, :, 0]
  FlowY = flowData[:, :, 1]
  return FlowX, FlowY

def flowToColor( U, V, maxFlow=None):
  H,W = U.shape
  UNKTHR = 1e9
  idsUNK = (np.abs(U) > UNKTHR) | (np.abs(V) > UNKTHR)
  U[ idsUNK] = 0
  V[ idsUNK] = 0

  MAXF = 999
  MINF = -999
  U = np.maximum( U, MINF)
  V = np.maximum( V, MINF)
  U = np.minimum( U, MAXF)
  V = np.minimum( V, MAXF)

  rad = np.sqrt( U**2 + V**2 )
  maxrad = rad.max()

  if maxFlow is not None:
    maxrad = maxFlow

  U = U/(maxrad + 1e-13)
  V = V/(maxrad + 1e-13)

  Img = computeColor( U,V)
  idsUNK= np.dstack( [idsUNK,idsUNK,idsUNK] )
  Img[idsUNK] = 0
  Img[:,:,0] = Img[::-1,:,0]
  Img[:,:,1] = Img[::-1,:,1]
  Img[:,:,2] = Img[::-1,:,2]
  Im = Image.fromarray( Img.astype('uint8')).convert('RGBA')
  return Im, Img

def computeColor( U, V):
  cwheel = makeColorwheel()
  nColors = cwheel.shape[0]
  print nColors
  
  #U = U.astype('float32')
  #V= V.astype('float32')
  rad = np.sqrt( U**2 + V**2 )
  a = np.arctan2( -V, -U )/np.pi
  fk = (a+1)/2.0*(nColors-1)  # map (-1,+1) to (0,1,2,...nC-1)
  k0 = np.uint32( np.floor( fk) )

  print a.min(), a.max()
  print k0.min(), k0.max()
  k1 = np.uint32( k0+1 )
  k1[ k1 == nColors] = 0
  f = fk - k0

  img = np.zeros( (U.shape[0], U.shape[1], 3) )
  for cid in xrange( 3 ):
    tmp = cwheel[ :, cid]
    c0  = tmp[k0]/255
    c1  = tmp[k1]/255
    col = (1-f)*c0 + f*c1

    idx = rad <=1
    col[ idx ] = 1-rad[idx] *(1-col[idx] ) # increase sat
    col[ ~idx] = col[~idx]*0.75
    img[:,:,cid] = np.floor( 255*col )
  return img

def makeColorwheel():
  RY = 15
  YG = 6
  GC = 4
  CB = 11
  BM = 13
  MR = 6

  nCols = RY + YG + GC + CB + BM + MR
  cwheel = np.zeros( (nCols,3) )
  
  RYvals = np.floor( 255*np.arange(0,RY)/RY )
  YGvals = np.floor( 255*np.arange(0,YG)/YG )
  GCvals = np.floor( 255*np.arange(0,GC)/GC )
  CBvals = np.floor( 255*np.arange(0,CB)/CB )
  BMvals = np.floor( 255*np.arange(0,BM)/BM )
  MRvals = np.floor( 255*np.arange(0,MR)/MR )

  col=0
  
  cwheel[:RY,0] = 255
  cwheel[:RY,1] = RYvals
  col += RY

  cwheel[col:col+YG,0] = 255 - YGvals
  cwheel[col:col+YG,1] = 255
  col += YG

  cwheel[col:col+GC,1] = 255
  cwheel[col:col+GC,2] = GCvals
  col += GC

  cwheel[col:col+CB,1] = 255 - CBvals
  cwheel[col:col+CB,2] = 255
  col += CB

  cwheel[col:col+BM,2] = 255
  cwheel[col:col+BM,0] = BMvals
  col += BM

  cwheel[col:col+MR,2] = 255 - MRvals
  cwheel[col:col+MR,0] = 255
  col += MR

  return cwheel
