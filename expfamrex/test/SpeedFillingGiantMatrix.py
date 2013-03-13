'''

Directly allocating a giant array and filling it in chunk by chunk
   is ~2x slower than just building up chunks in a list, and then calling vstack on that list

Tstart, Tstop = GetManyTs( 100, 3000 )
-------------------------------------------------- D=50
In [118]: timeit FillArray( Tstart, Tstop, 50)
1 loops, best of 3: 816 ms per loop

In [119]: timeit VStackArray( Tstart, Tstop, 50)
1 loops, best of 3: 429 ms per loop

-------------------------------------------------- D=500
In [137]: timeit FillArray( Tstart, Tstop, 500)
1 loops, best of 3: 7.56 s per loop

In [138]: timeit VStackArray( Tstart, Tstop, 500)
1 loops, best of 3: 4.24 s per loop


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
        rowIDs = range(Tstart[ii],Tstop[ii])
        Xnew = np.random.rand( len(rowIDs), D)
        Xall[rowIDs] = Xnew
    return Xall

def FillArray( Tstart, Tstop, D ):
    np.random.seed(D)
    Xall = np.zeros( (Tstop[-1], D) )
    for ii in xrange(len(Tstart)):
        rowIDs = xrange(Tstart[ii],Tstop[ii])
        Xnew = np.random.rand( len(rowIDs), D)
        Xall[rowIDs] = Xnew
    return Xall

def VStackArray( Tstart, Tstop,D ):
    np.random.seed(D)
    Xall = list()
    for ii in xrange( len(Tstart)):
        rowIDs = xrange( Tstart[ii], Tstop[ii])
        Xnew = np.random.rand( len(rowIDs), D)
        Xall.append( Xnew)
    return np.vstack( Xall )

