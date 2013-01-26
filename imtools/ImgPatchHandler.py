#!/usr/bin/python

import numpy as np
import scipy.misc

POptsDEFAULT = dict( border=0, Lx=8, Ly=8, dLx=4, dLy=4)

class ImgPatchHandler( object ):
  '''
     Class handles operations on a single image/video frame
         including 
                 + patch extraction
                 + patch visualization
  '''

  def __init__( self, img,  POpts=POptsDEFAULT, doGray=True ):
    '''
      >>> h = ImgPatchHandler( np.zeros( (5,2) ) )
      >>> print h.H, h.W
      5 2
      >>> h = ImgPatchHandler( np.zeros((300,100))  )
      >>> print h.H, h.W
      300 100
    '''

    if type(img) == str:
      img = scipy.misc.imread( img, flatten=doGray )
    else:
      assert type(img) == np.ndarray

    assert img.ndim ==2, 'Image must be grayscale (for now)'
    self.img = img
    
    self.H = self.img.shape[0]
    self.W = self.img.shape[1]

    for key in POptsDEFAULT:
      if key not in POpts:
        POpts[key] = POptsDEFAULT[key]
    self.POpts = POpts
    

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
      >>> h = ImgPatchHandler( A, POpts=POpts )
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

  def slicePatch( self, xs, ys ):
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
      >>> h = ImgPatchHandler( A, POpts=POpts )
      >>> S = h.slicePatch( [0,1,2], [1,2] )
      >>> print S
      [[ 1.  1.  2.]
       [ 2.  2.  1.]]
      >>> Sbad = h.slicePatch( [0,1,2], [4] ) # try to access out of bounds y
      Traceback (most recent call last):
      ... 
      IndexError: index out of range for array
    '''
    return self.img.take( ys, axis=0).take( xs, axis=1)

  def slicePatchXY( self, X, Y, xs, ys ):
    return X.take( ys, axis=0).take( xs, axis=1), Y.take( ys, axis=0).take( xs, axis=1) 

  def getNumPatches(self):
    xs,ys = self.getPatchCtrs()
    return len(xs)*len(ys)

  def FlatPatchGenerator( self ):
    '''
      Returns Lx*Ly  by D patches :  2D numpy arrays
      >>> POpts=dict( border=0, Lx=2, Ly=2, dLx=2, dLy=2, Nyc=1, Nxc=1 )
      >>> A1 = np.ones( (2,2) )
      >>> A2 = 2*np.ones( (2,2) )
      >>> A = np.vstack( [np.hstack( [A1, A2] ), np.hstack([A2,A1]) ] )
      >>> h = ImgPatchHandler( A, POpts=POpts )
      >>> Patches = [P for P in h.FlatPatchGenerator()]
      >>> print Patches[0]
      [ 1.  1.  1.  1.]
      >>> print Patches[1]
      [ 2.  2.  2.  2.]
      '''
    xs,ys = self.getPatchCtrs()
    dxs, dys = self.getPatchRelIDs()
    for y in ys:
      for x in xs:
        Patch = self.slicePatch( x+dxs, y+dys )
        yield Patch.reshape( (self.POpts['Lx']*self.POpts['Ly'] ) )

  def PatchGenerator( self ):
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
      >>> h = ImgPatchHandler( A, POpts=POpts )
      >>> Patches = [P for P in h.PatchGenerator()]
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
    for y in ys:
      for x in xs:
        Patch = self.slicePatch( x+dxs, y+dys )
        yield Patch

  def getMagAngleFromXY( self, X, Y ):
    '''
      >>> np.set_printoptions( precision=3 )
      >>> X,Y = np.meshgrid( np.linspace(-1,1,3), np.linspace(-1,1,3) )
      >>> print X
      [[-1.  0.  1.]
       [-1.  0.  1.]
       [-1.  0.  1.]]
      >>> h = ImgPatchHandler( np.eye(1) )
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

  def get_img_with_patch_borders( self ):
    aimg = self.img.copy()
    xCtrs,yCtrs  = self.getPatchCtrs()
    rx = self.POpts['Lx']/self.POpts['dLx']
    ry = self.POpts['Ly']/self.POpts['dLy']
    xCtrs = xCtrs[::rx]
    yCtrs = yCtrs[::ry]
    for x in xCtrs:
      for y in yCtrs:
        xLO = x - self.POpts['Lx']/2
        xHI = x + self.POpts['Lx']/2-1
        yLO = y - self.POpts['Ly']/2
        yHI = y + self.POpts['Ly']/2-1
        xs = np.arange( xLO, xHI+1 )
        ys = np.arange( yLO, yHI+1 )
        aimg[ ys, xLO] = 0
        aimg[ ys, xHI] = 0
        aimg[ yLO, xs] = 0
        aimg[ yHI, xs] = 0
    return aimg    

if __name__ == "__main__":
  import doctest
  doctest.testmod()
