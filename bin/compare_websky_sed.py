from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap
import numpy as np
import os,sys
from soapack import interfaces as sints
import healpy as hp
from tilec import fg as tfg


afreqs,anis = np.loadtxt(os.environ['WORK'] + "/cib_anis.txt",unpack=True)
mfreqs,ms = np.loadtxt("cib_monopoles.txt",unpack=True)

#afreqs,ms,anis = np.loadtxt(os.environ['WORK'] + "/new_cib.txt",unpack=True)
#mfreqs = afreqs

anis = np.sqrt(anis)/np.sqrt(anis[np.isclose(afreqs,100)])
ms = ms/ms[np.isclose(mfreqs,100)]

print(anis)
print(ms)





fs = np.geomspace(10,2000,1000)
th = tfg.get_mix(fs, 'CIB_Jysr')/tfg.get_mix(100, 'CIB_Jysr')
p =  {'Tdust_CIB': 13.6,'beta_CIB': 1.4,'nu0_CIB_ghz': 353.0}
th2 = tfg.get_mix(fs, 'CIB_Jysr',param_dict_override=p)/tfg.get_mix(100, 'CIB_Jysr',param_dict_override=p)

pl = io.Plotter(xyscale='loglog',xlabel='f',ylabel='m/m0')
pl._ax.scatter(afreqs,anis,color="C0",label='anis')
pl._ax.scatter(mfreqs,ms,color="C1",label='mono')
pl.add(fs,th,color="C2")
pl.add(fs,th2,color="C3")
pl.legend()
pl._ax.set_xlim(10,2000)
pl.done("comp_webskycib.png")

