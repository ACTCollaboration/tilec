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


l = [70,143,217,353,545,857]
freqs = [100] + l
fs = ["/scratch/r/rbond/msyriac/data/sims/websky/new_cib/cib_ns4096_nu%s.fits" % str(x).zfill(4) for x in  freqs]
f0 = 100.

acls = []

anis = []
monos = []



for i,f in enumerate(fs):

    freq = freqs[i]
    print(freq)
    imap = hp.read_map(f)
    _,mono = hp.remove_monopole(imap,fitval=True)

    ialm = hp.map2alm(imap)
    cls = hp.alm2cl(ialm)
    ells = np.arange(len(cls))
    acls.append(cls)

    anis.append(cls[np.logical_and(ells>4000,ells<6000)].mean())
    monos.append(mono)

    

io.save_cols(os.environ['WORK'] + "/new_cib.txt",(freqs,monos,anis))
