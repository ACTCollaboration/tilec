from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap,reproject
import numpy as np
import os,sys
from soapack import interfaces as sints
import healpy as hp


import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("region", type=str,help='Region name.')
parser.add_argument("mtype", type=str,help='phi or def or kappa.')
parser.add_argument("--planck", action='store_true',help='Planck instead of ACT.')
args = parser.parse_args()


"""
Saves filtered maps. Run kplot2.py after.
"""

f = lambda x: enmap.fft(x,normalize='phys')
p = lambda x: (x*x.conj()).real

region = args.region
mtype = args.mtype
planck = args.planck


if planck: pstr = "_planck"
else : pstr = ""

mask = enmap.downgrade(sints.get_act_mr3_crosslinked_mask(region),4)
w4 = np.mean(mask**4)

if not(planck):
    kname = "/scratch/r/rbond/msyriac/data/act/omar/03102019/realKappaCoadd_s14&15_%sNew.fits" % region
    imap = enmap.downgrade(enmap.read_map(kname),4)
else:
    hname = "/scratch/r/rbond/msyriac/data/planck/data/pr3/COM_Lensing_4096_R3.00/MV/dat_klm.fits"
    halm = hp.read_alm(hname)
    imap = reproject.enmap_from_healpix(halm, mask.shape, mask.wcs, ncomp=1, unit=1, lmax=0,
                                        rot="gal,equ", first=0, is_alm=True, return_alm=False)
    

# Filtering
modlmap = imap.modlmap()
p2d = p(f(imap))/w4
bin_edges = np.arange(20,3000,20)
binner = stats.bin2D(modlmap,bin_edges)
theory = cosmology.default_theory()
clkk2d = theory.gCl('kk',modlmap)
cents,t1d = binner.bin(p2d)
w1d = binner.bin(clkk2d)[1]/t1d
w1d[cents<80] = 0

if mtype=='kappa':
    dfact = 1
elif mtype=='phi':
    dfact = modlmap**2
elif mtype=='def':
    dfact = modlmap

w2d = maps.interp(cents,w1d)(modlmap) / dfact
w2d[~np.isfinite(w2d)] = 0
fmap = maps.filter_map(imap,w2d)

io.hplot(fmap*mask,'fmap_%s_%s%s' % (mtype,region,pstr),color='gray')
enmap.write_map("fmap_%s_%s%s.fits" % (mtype,region,pstr),fmap*mask)

hname = "/scratch/r/rbond/msyriac/data/planck/data/pr3/COM_CompMap_CIB-GNILC-F545_2048_R2.00.fits"
hmap = hp.read_map(hname)
io.mollview(hmap,"hmapmoll_%s%s.png" % (region,pstr),coord=['G','C'])
hmap = reproject.enmap_from_healpix(hname, mask.shape, mask.wcs, ncomp=1, unit=1, lmax=0,
                                    rot="gal,equ", first=0, is_alm=False, return_alm=False)
io.hplot(hmap,'cibmap_%s%s' % (region,pstr),color='planck')
hmap = maps.filter_map(hmap*mask,w2d)
io.hplot(hmap*mask,'hmap_%s_%s%s' % (mtype,region,pstr),color='gray')
enmap.write_map("hmap_%s_%s%s.fits" % (mtype,region,pstr),hmap*mask)



