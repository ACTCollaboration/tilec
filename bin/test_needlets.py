from __future__ import print_function
from tilec import needlets
from orphics import io
from pixell import enmap
import numpy as np
import os,sys

lmax = 30000
fwhms = np.array([600., 300., 120., 60., 30., 15., 10., 7.5, 5.,4.,3.,2.,1.0])
filters = needlets.gaussian_needlets(lmax,fwhms)
print(filters.shape)

ls = np.arange(filters.shape[1])

pl = io.Plotter(xyscale='loglin',xlabel='l',ylabel='f')
for i in range(filters.shape[0]): pl.add(ls[2:],filters[i,2:],label=str(i))
pl.add(ls[2:],(filters[:,2:]**2.).sum(axis=0),color='k')
pl.done('filters.png')
