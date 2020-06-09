import numpy as np
import os,sys
cimport numpy as cnp
from libc.stdlib cimport malloc, free
cimport scipy.linalg.cython_lapack as cl

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def map_cinv(cnp.ndarray[DTYPE_t, ndim=4, mode='c'] c):
    assert c.dtype == DTYPE
    cdef int Np = c.shape[0]
    cdef int N = c.shape[2]
    cdef cnp.ndarray[DTYPE_t, ndim=4] cov = np.empty([Np,Np,N,N], dtype=DTYPE)   
    cdef cnp.ndarray[DTYPE_t, ndim=4] cinv = np.empty([Np,Np,N,N], dtype=DTYPE)  
    for ip in range(Np):
        for jp in range(Np):
            for i in range(N):
                for j in range(i,N):
                    cov[ip,jp,i,j] = c[ip,jp,i,j]
                    cov[ip,jp,j,i] = c[ip,jp,j,i]
            cinv[ip,jp] = invert(cov[ip,jp])
    return cinv

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def invert(cnp.ndarray[DTYPE_t, ndim=2] array):

    cdef  int rows = array.shape[0]
    cdef   int cols = array.shape[1]
    cdef  int info = 0
    if cols !=rows:
        return array,1,"not a square matrix"

    cdef int* ipiv = <int *> malloc(rows * sizeof(int))
    if not ipiv:
        raise MemoryError()

    cl.dgetrf(&cols,&rows,&array[0,0],&rows,ipiv,&info)
    if info !=0:
        free(ipiv)
        return array,info,"dgetrf failed, INFO="+str(info)
    #workspace query
    cdef double workl
    cdef int lwork=-1
    cl.dgetri(&cols,&array[0,0],&rows,ipiv,&workl,&lwork,&info)
    if info !=0:
        free(ipiv)
        return array,info,"dgetri failed, workspace query, INFO="+str(info)
    #allocation workspace
    lwork= int(workl)
    cdef double* work = <double *> malloc(lwork * sizeof(double))
    if not work:
        raise MemoryError()

    cl.dgetri(&cols,&array[0,0],&rows,ipiv,work,&lwork,&info)
    if info !=0:
        free(ipiv)
        free(work)
        return array,info,"dgetri failed, INFO="+str(info)

    free(ipiv)
    free(work)

    return array
