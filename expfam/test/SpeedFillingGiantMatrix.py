'''

Directly allocating a giant array and filling it in chunk by chunk
   is very slow (~2x) when using a "full range" indexing
   
Instead, if we just provide start and stop bounds, this improves tremendously.

Just start+stop bounds is even faster than using vstack

[on macbook pro late 2011]
>>> Tstart, Tstop = GetManyTs( 4000, 300 )

>>> timeit FillArrayRange( Tstart, Tstop, 50)
1 loops, best of 3: 2.23 s per loop

>>> timeit FillArray( Tstart, Tstop, 50)
1 loops, best of 3: 1.09 s per loop

>>> timeit VStackArray( Tstart, Tstop,50)
1 loops, best of 3: 1.29 s per loop

'''

import numpy as np

def GetManyTs( Nseq, T):
    Tstart = list()
    Tstop = list()
    t=0
    for ii in xrange(Nseq):
      Tstart.append( t )
      Tstop.append( t+T )
      t += T
    return Tstart, Tstop

def FillArrayRange( Tstart, Tstop, D ):
    ''' this isn't any different than just FillArray
    '''
    np.random.seed(D)
    Xall = np.empty( (Tstop[-1], D) )
    for ii in xrange(len(Tstart)):
        rowIDs = xrange(Tstart[ii],Tstop[ii])
        Xnew = np.random.rand( len(rowIDs), D)
        Xall[rowIDs] = Xnew
    return Xall

def FillArray( Tstart, Tstop, D ):
    np.random.seed(D)
    Xall = np.zeros( (Tstop[-1], D) )
    for ii in xrange(len(Tstart)):
        Xnew = np.random.rand( Tstop[ii]-Tstart[ii], D)
        Xall[ Tstart[ii]:Tstop[ii] ] = Xnew
    return Xall

def VStackArray( Tstart, Tstop,D ):
    np.random.seed(D)
    Xall = list()
    for ii in xrange( len(Tstart)):
        rowIDs = xrange( Tstart[ii], Tstop[ii])
        Xnew = np.random.rand( len(rowIDs), D)
        Xall.append( Xnew)
    return np.vstack( Xall )

