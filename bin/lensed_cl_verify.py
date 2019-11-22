from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap
import numpy as np
import os,sys
from soapack import interfaces as sints
from tilec import utils as tutils,covtools,pipeline
import symlens
from enlib import bench

"""
compares binned theory to input CMB, I think
"""

region = 'deep56'
w = os.environ['WORK'] + "/"
deg_taper = 3.

tdir = '/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919'
mask = enmap.read_map(tutils.get_generic_fname(tdir,region,'cmb',sim_index=None,mask=True))
mask,_ = maps.get_taper_deg(mask.shape,mask.wcs,deg_taper)


shape,wcs = mask.shape,mask.wcs
theory = cosmology.default_theory()
modlmap = mask.modlmap()
lcl = theory.lCl('TT',modlmap)

Nsims = 200
comm,rank,my_tasks = mpi.distribute(Nsims)

kellmin = 450
kellmax = 4500
bin_edges = np.arange(kellmin,kellmax,40)
binner = stats.bin2D(modlmap,bin_edges)
w2 = np.mean(mask**2)

s = stats.Stats(comm)
fc = maps.FourierCalc(shape,wcs)
p = lambda x,y: fc.f2power(x,y)
cents,i1d = binner.bin(theory.lCl('TT',modlmap))

for task in my_tasks:
    print(rank,task)
    index = task
    imap = pipeline.get_input('CMB',0,index,shape,wcs) * mask
    kmap = fc.power2d(imap)[1]


    cents,c1d = binner.bin(p(kmap,kmap)/w2)

    d1d = (c1d-i1d)/i1d

    s.add_to_stats('d1d',d1d)

s.get_stats()

if rank==0:
    d1d = s.stats['d1d']['mean']
    ed1d = s.stats['d1d']['errmean']

    pl = io.Plotter(xlabel='l',ylabel='r')
    pl.add_err(cents,d1d,yerr=ed1d)
    pl.hline(y=0)
    pl.done(w + "/cmbcldiffverify.png")

