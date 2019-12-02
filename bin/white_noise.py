from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from soapack import interfaces as sints
import tilec.fg as tfg
import tilec.utils as tutils

solution = 'cmb'
tdir = '/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919'
cversion = 'joint'

f = lambda x: enmap.fft(x,normalize='phys')

bin_edges = np.arange(500,15000,100)

pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='C')

dm = sints.ACTmr3()

for region in ['deep56','boss']:
    mask = sints.get_act_mr3_crosslinked_mask(region)
    sfile = tutils.get_generic_fname(tdir,region,solution,None,cversion)
    cfile1 = tutils.get_generic_fname(tdir,region,solution,'tsz',cversion)
    cfile2 = tutils.get_generic_fname(tdir,region,solution,'cib',cversion)


    w2 = np.mean(mask**2.)
    modlmap = mask.modlmap()
    binner = stats.bin2D(modlmap,bin_edges)

    p = lambda x: (f(x)*f(x).conj()).real/w2

    #prat = (dm.get_beam(modlmap,'s15',region,'pa3_f150',sanitize=True)/maps.gauss_beam(modlmap,1.6))**2.
    prat = (maps.gauss_beam(modlmap,1.4)/maps.gauss_beam(modlmap,1.6))**2.
    prat = 1

    cents,p1d = binner.bin(p(enmap.read_map(sfile))*prat)
    pl.add(cents,p1d,label=region)
    print(np.sqrt(p1d[-1])*60.*180./np.pi)

    #prat = (dm.get_beam(modlmap,'s15',region,'pa3_f150',sanitize=True)/maps.gauss_beam(modlmap,2.4))**2.
    prat = (maps.gauss_beam(modlmap,2.1)/maps.gauss_beam(modlmap,2.4))**2.
    prat = 1

    cents,p1d = binner.bin(p(enmap.read_map(cfile1))*prat)
    pl.add(cents,p1d,label=region + "-tsz")
    print(np.sqrt(p1d[-1])*60.*180./np.pi)

    prat = (maps.gauss_beam(modlmap,1.9)/maps.gauss_beam(modlmap,2.4))**2.
    prat = 1

    cents,p1d = binner.bin(p(enmap.read_map(cfile2))*prat)
    pl.add(cents,p1d,label=region + "-cib")
    print(np.sqrt(p1d[-1])*60.*180./np.pi)

pl.done("white_noise.png")
