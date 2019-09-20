from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from soapack import interfaces as sints
from tilec import utils as tutils,covtools


ells = np.arange(60,3000,1)
lc = cosmology.LimberCosmology(skipCls=True,pickling=False,numz=1000,kmax=1.47,nonlinear=True,skipPower=False,zmax=1100.,low_acc=True,skip_growth=True)
lc.addStepNz("g",0.4,0.7,bias=2.0)
#lc.addStepNz("g",0.2,0.9,bias=1.6)
lc.generateCls(ells )
clkk = lc.getCl("cmb","cmb")
clkg = lc.getCl("cmb","g")
clgg = lc.getCl("g","g")

#theory = cosmology.default_theory()
#clkk = theory.gCl('kk',ells)

ledges = np.arange(80,1000,40)

for region in ['deep56','boss']:
    dmask = sints.get_act_mr3_crosslinked_mask(region=region)
    bmask = maps.binary_mask(dmask,0.95)
    fsky = dmask.area()/np.pi/4. * (bmask.sum()/bmask.size)

    pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='C')
    pl.add(ells,clkk)
    ls,nls = np.loadtxt("lensing_noise_act_planck_ilc_all_%s.txt" % region,unpack=True)
    pl.add(ls,nls)
    oells,onls = np.loadtxt("cl_K_ilc_noszK_ilc_noszs14&15_%s.txt" % region , unpack=True,usecols=[0,1])
    pl.add(oells,onls-maps.interp(ells,clkk)(oells),ls='--')
    pl.done("lensnoise_%s.png" % region)

    lf = cosmology.LensForecast()
    lf.loadKK(ells,clkk,ls,nls)
    lf.loadKG(ells,clkg)
    lf.loadGG(ells,clgg,ngal=0.026)
    #lf.loadGG(ells,clgg,ngal=0.1)
    snr = lf.sn(ledges,fsky,'kk')[0]
    snr_kg = lf.sn(ledges,fsky,'kg')[0]
    print(fsky*41252,snr,snr_kg)
