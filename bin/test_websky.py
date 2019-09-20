from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap
import numpy as np
import os,sys
from soapack import interfaces as sints
import healpy as hp
from tilec import fg as tfg
import glob

for freq in [30,44,70,100,143,217,353,545,857]:
    v1 = tfg.get_mix_bandpassed([glob.glob('data/*_BANDPASS_F%s_reformat.txt' % str(freq).zfill(3))[0]], 'CIB', normalize_cib=False)
    v2 = tfg.get_mix(freq, 'CIB_Jysr')
    v3 = tfg.get_mix(freq, 'CIB')
    print(v1/1e26,v2/1e26,v3/1e26)
sys.exit()

# print(tfg.ItoDeltaT(545))
# sys.exit()
lmax = 6000 #8192*3
cmb_set = 0
nu_ghz = 92

def get_cls(alms):
    calm = maps.change_alm_lmax(alms, lmax)        
    cls = hp.alm2cl(calm)
    ls = np.arange(len(cls))
    return ls,cls


rpath = sints.dconfig['actsims']['websky_path'] + "/" 

pl = io.Plotter(xyscale='loglog',scalefn=lambda x: x**2./2./np.pi,xlabel='l',ylabel='D')

ialm = hp.read_alm(rpath + "lensed_alm_seed%d.fits" % cmb_set,hdu=(1,2,3))
for i in range(1): 
    ls,cls = get_cls(ialm[i])
    pl.add(ls,cls,label='cmb %d' % i)


tconversion = 1 # !!!

ialm = hp.read_alm(rpath + "alms/ksz_alm.fits" )
ls,cls = get_cls(ialm)
pl.add(ls,cls,label='ksz')



ialm = hp.read_alm(rpath + "alms/ksz_patchy_alm.fits" )
ls,cls = get_cls(ialm)
pl.add(ls,cls,label='ksz patchy')

ialm = hp.read_alm(rpath + "alms/cib_%s_alm.fits" % (str(nu_ghz).zfill(4)) ) * 1e6 * tfg.ItoDeltaT(nu_ghz) / 1e26
ls,cls = get_cls(ialm)
pl.add(ls,cls,label='cib %d' % nu_ghz)


tcon = tfg.get_mix(nu_ghz, 'tSZ')
ialm = hp.read_alm(rpath + "alms/tsz_8192_alm.fits" ) * tcon
ls,cls = get_cls(ialm)
pl.add(ls,cls,label='tsz')

pl.done("websky.png")
sys.exit()
