from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap
import numpy as np
import os,sys
from soapack import interfaces as sints

def get_coadd(imaps,wts,axis):
    # sum(w*m)/sum(w)
    twt = np.sum(wts,axis=axis)
    retmap = np.sum(wts*imaps,axis=axis)/twt
    retmap[~np.isfinite(retmap)] = 0
    return retmap,twt

def get_npol(array):
    if array=='545' or array=='857':return 1
    else: return 3

mask = sints.get_act_mr3_crosslinked_mask('deep56',version='180323')
dm = sints.PlanckHybrid(region=mask)
bin_edges = np.arange(30,6000,40)

p1ds = {}
for array in dm.arrays:
    splits = dm.get_splits(array,srcfree=False)[0]
    ivars = dm.get_splits_ivar(array)[0]
    coadd,wt = get_coadd(splits,ivars,axis=0)
    npol = get_npol(array)
    for i in range(npol):
        cents,p1d = maps.binned_power(coadd[i],bin_edges=bin_edges,mask=mask)
        p1ds[array+str(i)] = p1d.copy()
        mivar = wt[i].mean()

    print(array,mivar)

for i in range(3):
    pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='C')
    for array in dm.arrays:
        npol = get_npol(array)
        if i<npol:
            pl.add(cents,p1ds[array+str(i)],label=array)
    pl.done("powers%d.png" % i)
