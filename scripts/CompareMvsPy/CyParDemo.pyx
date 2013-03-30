# CyParDemo.pyx
#   from http://stackoverflow.com/questions/13068760/parallelise-python-loop-with-numpy-arrays-and-shared-memory
from cython.parallel cimport prange

import numpy as np

def foo():
    cdef int i, j, n

    x = np.random.rand( 20000, 2000 )

    n = x.shape[0]
    for i in prange(n, nogil=True):
        with gil:
            np.cos(x[i,:])

    return x
