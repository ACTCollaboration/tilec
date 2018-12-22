from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from tilec import covtools,utils as tutils

"""
Script that makes TILe-C propaganda plots
of 2D noise powers.
"""

nsplits = 4
theory = cosmology.default_theory(lpad=40000)

shape,wcs = maps.rect_geometry(width_deg=10.,px_res_arcmin=0.5)
modlmap = enmap.modlmap(shape,wcs)

rms = 1.
lknee = 2500
alpha = -4
ps_noise = covtools.noise_average(covtools.get_anisotropic_noise(shape,wcs,rms,lknee,alpha,template_file=None,tmin=0,tmax=100),(16,16),radial_fit=True,bin_annulus=80)[0]
ps_noise[modlmap<20] = 0
ps_signal = enmap.enmap(theory.lCl('TT',modlmap),wcs)

mgen = maps.MapGen(shape,wcs,ps_signal[None,None])
ngen = maps.MapGen(shape,wcs,ps_noise[None,None]*nsplits)

cmb = mgen.get_map(seed=(1,0))
observed = []
for i in range(nsplits):
    nreal = ngen.get_map(seed=(2,i))
    observed.append(cmb+nreal)

kmaps = []
fc = maps.FourierCalc(shape,wcs)
for i in range(nsplits):
    kmaps.append(fc.power2d(observed[i])[1])
    

io.hplot(maps.ftrans(ps_noise),"noise_orig",grid=False)
io.hplot(maps.ftrans(ps_signal),"signal_orig",grid=False)
io.hplot(maps.ftrans(ps_noise+ps_signal),"total_orig",grid=False)

scov,ncov,_ = tutils.ncalc(np.array(kmaps)[None],0,0,fc)
io.hplot(maps.ftrans(ncov),"noise_real",grid=False)
io.hplot(maps.ftrans(scov),"signal_real",grid=False)

dscov = covtools.signal_average(scov,bin_width=80,kind=0)
dncov0 = covtools.signal_average(ncov,bin_width=80,kind=0)
dncov = covtools.noise_average(ncov,dfact=(16,16),lmin=300,lmax=8000,wnoise_annulus=500,bin_annulus=20,
                  lknee_guess=3000,alpha_guess=-4,nparams=None,modlmap=None,
                  verbose=False,method="fft",radial_fit=True,
                  oshape=None,upsample=True)[0]


    
io.hplot(maps.ftrans(dscov),"signal_smoothed",grid=False)
io.hplot(maps.ftrans(dncov0),"noise_smoothed_iso",grid=False)
io.hplot(maps.ftrans(dncov),"noise_smoothed",grid=False)
