#!/usr/bin/python

import numpy as np
import cv
import CVUtils as CVU

POptsDEFAULT = dict( border=10, Lx=20, Ly=20, dLx=10, dLy=10,   Lt=8, Nyc=1,Nxc=1,Ntc=1)

class FrameHandle( object ):
  '''
     Class handles operations on a single frame of video
         including 
                 + patch extraction
                 + descriptor extraction
                 + descriptor visualization
  '''

  def __init__( self, imCur, frameID=0, scaleID=0, POpts=POptsDEFAULT, vidFileName=None, imNext=None ):
    '''
      >>> h = FrameHandle( np.zeros( (5,2) ) )
      >>> print h.H, h.W
      5 2
      >>> Q = cv.CreateImage( (100,300), cv.IPL_DEPTH_32F, 3 )
      >>> h = FrameHandle( Q )
      >>> print h.H, h.W
      300 100
    '''
    self.frameID = frameID
    self.scaleID = scaleID

    if type(imCur) == str:
      self.CVIm = cv.LoadImage( imCur )
      self.Im   = CVU.CVMat2NPArray( self.CVIm )
    elif type(imCur) == np.ndarray:
      imCur = np.float32( imCur )
      self.CVIm = cv.GetImage( CVU.NPArray2CVMat( imCur ) )
      self.Im   = imCur
    elif type(imCur) == cv.iplimage or type(imCur) == cv.cvmat:
      self.CVIm = imCur
      self.Im   = CVU.CVMat2NPArray( imCur )

    self.CVImGray = CVU.ConvertToGrayscale( self.CVIm )
    self.H = self.Im.shape[0]
    self.W = self.Im.shape[1]

    for key in POptsDEFAULT:
      if key not in POpts:
        POpts[key] = POptsDEFAULT[key]

    self.POpts = POpts

  def getPatchDescriptors( self ):
    pass

  def getOpticalFlow( self, methName ):
    '''
       Returns:
          Fx :  H x W numpy array
          Fy :  H x W numpy array
    '''
    pass

  def getSpatialGradient( self ):
    '''
       Returns:
          Gx :  H x W numpy array
          Gy :  H x W numpy array
    '''
    gradX = cv.CreateImage( cv.GetSize(self.CVImGray), 32, 1)
    gradY = cv.CreateImage( cv.GetSize(self.CVImGray), 32, 1)
    cv.Sobel(self.CVImGray, gradX, 1, 0, 1)
    cv.Sobel(self.CVImGray, gradY, 0, 1, 1)
    gradXarr = CVU.CVMat2NPArray( gradX )
    gradYarr = CVU.CVMat2NPArray( gradY )
    return gradXarr, gradYarr

  def getPatchRelIDs( self ):
    dxs = np.arange(-(self.POpts['Lx']/2), self.POpts['Lx']/2 )
    dys = np.arange(-(self.POpts['Ly']/2), self.POpts['Ly']/2 ) 
    return np.int8(dxs), np.int8(dys)


  def getPatchCtrs( self ):
    '''
      >>> POpts=dict( border=0, Lx=2, Ly=2, dLx=2, dLy=2, Nyc=1, Nxc=1 )
      >>> A1 = np.ones( (2,2) )
      >>> A2 = 2*np.ones( (2,2) )
      >>> A = np.vstack( [np.hstack( [A1, A2] ), np.hstack([A2,A1]) ] )
      >>> print A
      [[ 1.  1.  2.  2.]
       [ 1.  1.  2.  2.]
       [ 2.  2.  1.  1.]
       [ 2.  2.  1.  1.]]
      >>> h = FrameHandle( A, POpts=POpts )
      >>> print h.getNumPatches() 
      4
      >>> xs,ys= h.getPatchCtrs()
      >>> print xs
      [1 3]
      >>> print ys
      [1 3]
    '''
    xs = np.arange( self.POpts['border']+self.POpts['Lx']/2, self.W-self.POpts['border'], self.POpts['dLx'] )
    ys = np.arange( self.POpts['border']+self.POpts['Ly']/2, self.H-self.POpts['border'], self.POpts['dLy'] )
    xs = xs[ (xs+self.POpts['Lx']/2-1) < self.W ]
    ys = ys[ (ys+self.POpts['Ly']/2-1) < self.H ]
    return xs, ys


  def getMagAngleFromXY( self, X, Y ):
    '''
      >>> np.set_printoptions( precision=3 )
      >>> X,Y = np.meshgrid( np.linspace(-1,1,3), np.linspace(-1,1,3) )
      >>> print X
      [[-1.  0.  1.]
       [-1.  0.  1.]
       [-1.  0.  1.]]
      >>> h = FrameHandle( np.eye(1) )
      >>> M,T = h.getMagAngleFromXY( X, Y )
      >>> print T
      [[ 225.  270.  315.]
       [ 180.    0.    0.]
       [ 135.   90.   45.]]
      >>> print M
      [[ 1.414  1.     1.414]
       [ 1.     0.     1.   ]
       [ 1.414  1.     1.414]]
    '''
    M = np.sqrt( X*X+Y*Y )
    T = 180.0/np.pi*np.arctan2(Y,X)
    badts=T<0
    T[badts] = 360+T[badts]
    return M,T

  def slicePatch( self, Im, xs, ys ):
    '''
      >>> POpts=dict( border=0, Lx=2, Ly=2, dLx=2, dLy=2, Nyc=1, Nxc=1 )
      >>> A1 = np.ones( (2,2) )
      >>> A2 = 2*np.ones( (2,2) )
      >>> A = np.vstack( [np.hstack( [A1, A2] ), np.hstack([A2,A1]) ] )
      >>> print A
      [[ 1.  1.  2.  2.]
       [ 1.  1.  2.  2.]
       [ 2.  2.  1.  1.]
       [ 2.  2.  1.  1.]]
      >>> h = FrameHandle( A, POpts=POpts )
      >>> S = h.slicePatch( A, [0,1,2], [1,2] )
      >>> print S
      [[ 1.  1.  2.]
       [ 2.  2.  1.]]
      >>> Sbad = h.slicePatch( A, [0,1,2], [4] ) # try to access out of bounds y
      Traceback (most recent call last):
      ... 
      IndexError: index out of range for array
    '''
    return Im.take( ys, axis=0).take( xs, axis=1)

  def slicePatchXY( self, X, Y, xs, ys ):
    return X.take( ys, axis=0).take( xs, axis=1), Y.take( ys, axis=0).take( xs, axis=1) 

  def getNumPatches(self):
    xs,ys = self.getPatchCtrs()
    return len(xs)*len(ys)

  def FlatPatchIterator( self, Im ):
    '''
      Returns Lx*Ly  by D patches :  2D numpy arrays
      >>> POpts=dict( border=0, Lx=2, Ly=2, dLx=2, dLy=2, Nyc=1, Nxc=1 )
      >>> A1 = np.ones( (2,2) )
      >>> A2 = 2*np.ones( (2,2) )
      >>> A = np.vstack( [np.hstack( [A1, A2] ), np.hstack([A2,A1]) ] )
      >>> A = np.dstack( [A, 10*A, 100*A] )
      >>> h = FrameHandle( A, POpts=POpts )
      >>> Patches = [P for pID,P in h.FlatPatchIterator(A)]
      >>> print Patches[0]
      [[   1.   10.  100.]
       [   1.   10.  100.]
       [   1.   10.  100.]
       [   1.   10.  100.]]
      >>> print Patches[1]
      [[   2.   20.  200.]
       [   2.   20.  200.]
       [   2.   20.  200.]
       [   2.   20.  200.]]
    '''
    if Im.ndim == 2:
      Im = Im[:,:,np.newaxis]
    xs,ys = self.getPatchCtrs()
    dxs, dys = self.getPatchRelIDs()
    pID = 0
    for y in ys:
      for x in xs:
        Patch = self.slicePatch( Im, x+dxs, y+dys )
        yield pID, Patch.reshape( (self.POpts['Lx']*self.POpts['Ly'], Im.shape[2]) )
        pID+=1

  def FlatPatchIteratorXY( self, X, Y ):
    if X.ndim == 2:
      X = X[:,:,np.newaxis]
    if Y.ndim == 2:
      Y = Y[:,:,np.newaxis]
    xs,ys = self.getPatchCtrs()
    dxs, dys = self.getPatchRelIDs()
    pID = 0
    N = self.POpts['Lx']*self.POpts['Ly']
    Dx = X.shape[2]
    Dy = Y.shape[2]
    for y in ys:
      for x in xs:
        pX,pY = self.slicePatchXY( X, Y, x+dxs, y+dys )
        yield pID, pX.reshape( (N,Dx) ), pY.reshape( (N,Dy) )
        pID+=1

  def PatchIterator( self, Im ):
    '''
      >>> POpts=dict( border=0, Lx=2, Ly=2, dLx=2, dLy=2, Nyc=1, Nxc=1 )
      >>> A1 = np.ones( (2,2) )
      >>> A2 = 2*np.ones( (2,2) )
      >>> A = np.vstack( [np.hstack( [A1, A2] ), np.hstack([A2,A1]) ] )
      >>> print A
      [[ 1.  1.  2.  2.]
       [ 1.  1.  2.  2.]
       [ 2.  2.  1.  1.]
       [ 2.  2.  1.  1.]]
      >>> h = FrameHandle( A, POpts=POpts )
      >>> Patches = [P for pID,P in h.PatchIterator(A)]
      >>> print Patches[0]
      [[ 1.  1.]
       [ 1.  1.]]
      >>> print Patches[1]
      [[ 2.  2.]
       [ 2.  2.]]
      >>> print Patches[-1]
      [[ 1.  1.]
       [ 1.  1.]]
    '''
    xs,ys = self.getPatchCtrs()
    dxs, dys = self.getPatchRelIDs()
    pID = 0
    for y in ys:
      for x in xs:
        Patch = self.slicePatch( Im, x+dxs, y+dys )
        yield pID, Patch
        pID+=1

  def PatchIteratorXY( self, X, Y ):
    xs,ys = self.getPatchCtrs()
    dxs, dys = self.getPatchRelIDs()
    pID = 0
    for y in ys:
      for x in xs:
        pX,pY = self.slicePatchXY( X, Y, x+dxs, y+dys )
        yield pID, pX, pY
        pID+=1

if __name__ == "__main__":
  import doctest
  doctest.testmod()
