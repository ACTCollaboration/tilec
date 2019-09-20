from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap
import numpy as np
import os,sys
from soapack import interfaces as sints
import healpy as hp
from tilec import fg as tfg
import glob
import random

rpath = sints.dconfig['actsims']['websky_path'] + "/" 


l = [146,224,353,857,40,545,26]
random.shuffle(l)
fs = [rpath + "alms/cib_%s_alm.fits" % str(x).zfill(4) for x in  [100] + l]
f0 = 100.

acls = []

anis = []
freqs = []

for i,f in enumerate(fs):
    freq = int(f.split('_')[1])
    #r = tfg.get_mix(f0, 'CIB')/tfg.get_mix(freq, 'CIB')
    r = 1 #tfg.get_mix(f0, 'CIB_Jysr')/tfg.get_mix(freq, 'CIB_Jysr')
    print(freq)
    #ialm = hp.read_alm(f) * 1e6 * tfg.ItoDeltaT(freq) / 1e26 * r
    ialm = hp.read_alm(f) * r 

    cls = hp.alm2cl(ialm)
    ells = np.arange(len(cls))
    acls.append(cls)

    anis.append(cls[np.logical_and(ells>4000,ells<6000)].mean()/r/r)
    freqs.append(freq)

    # pl = io.Plotter(xyscale='linlin',xlabel='l',ylabel='C/C0')
    # for j,cl in enumerate(acls):
    #     pl.add(ells,cl/acls[0],label=fs[j].split('_')[1])
    # pl.hline(y=1)
    # pl._ax.set_ylim(0,3)
    # pl.done("webskycib.png")
    

io.save_cols(os.environ['WORK'] + "/cib_anis.txt",(freqs,anis))

    

