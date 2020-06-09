from __future__ import print_function
import numpy as np
import os,sys
from tilec import cneedlet
from enlib import bench

Np = 200
shape = (Np,Np)
N = 40

a = np.random.random((N,N)).astype(np.float64)
c = np.dot(a,a.T)
c = np.repeat(c[None,...],Np,axis=0)
c = np.repeat(c[None,...],Np,axis=0)




with bench.show("vec"):
    cov = np.zeros((Np,Np,N,N),dtype=np.float64)
    cinv = np.zeros((Np,Np,N,N),dtype=np.float64)
    for i in range(N):
        for j in range(i,N):
            cov[...,i,j] = c[...,i,j]
            cov[...,j,i] = c[...,j,i]

    cinv = np.linalg.inv(cov)


ocov = cov.copy()
ocinv = cinv.copy()

with bench.show("cython"):
    ccinv = cneedlet.map_cinv(c)


# with bench.show("dumb"):
#     cov = np.zeros((Np,Np,N,N),dtype=np.float64)
#     cinv = np.zeros((Np,Np,N,N),dtype=np.float64)
#     for ip in range(Np):
#         for jp in range(Np):
#             for i in range(N):
#                 for j in range(i,N):
#                     cov[ip,jp,i,j] = c[ip,jp,i,j]
#                     cov[ip,jp,j,i] = c[ip,jp,j,i]
#             cinv[ip,jp] = np.linalg.inv(cov[ip,jp])

# assert np.all(np.isclose(cov,ocov))
# assert np.all(np.isclose(cinv,ocinv))
assert np.all(np.isclose(ccinv,ocinv))

