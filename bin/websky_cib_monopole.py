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
fs = [rpath + "cib_%s.fits" % str(x).zfill(4) for x in  [100] + l]
f0 = 100

acls = []

monos = {}
ms = []
freqs = []
for i,f in enumerate(fs):
    freq = int(f.split('_')[1].split('.')[0])
    # r = tfg.get_mix(f0, 'CIB_Jysr')/tfg.get_mix(freq, 'CIB_Jysr')

    imap = hp.read_map(f)
    _,mono = hp.remove_monopole(imap,fitval=True)
    monos[freq] = mono
    ms.append(mono)
    freqs.append(freq)

io.save_cols("cib_monopoles.txt",(freqs,ms))

pl = io.Plotter(xyscale='loglin',xlabel='f',ylabel='m/m0')
for freq in monos.keys():
    r = tfg.get_mix(f0, 'CIB_Jysr')/tfg.get_mix(freq, 'CIB_Jysr')/monos[f0]
    pl._ax.scatter(freq,monos[freq]*r,color="C0")
pl.hline(y=1)
pl.done("mono_webskycib.png")
    


    

