from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
from orphics import maps,io,cosmology,stats
from pixell import enmap,reproject
import numpy as np
import os,sys
import healpy as hp
from soapack import interfaces as sints
import tilec.fg as tfg
import tilec.utils as tutils
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

redo = True
fwhm = 5.

ppath = "/scratch/r/rbond/msyriac/data/planck/data/pr3/"

alm = hp.read_alm(ppath + 'COM_CMB_IQU-smica-nosz_2048_R3.00_full_alm.fits',hdu=1) * 1e6


method='_deprojects_comptony' 
methodp= 'nosz'
pl = io.Plotter(xyscale='linlog',xlabel='$\\ell$',
                ylabel='$\\ell^2 C^{\\rm{CMB},\\rm{tot}}_{\\ell}/2\\pi (\\mu K-\\rm{rad})^2 $',
                scalefn = lambda x: x**2./2./np.pi,ftsize=16,labsize=14)
for col,region in zip(['red','blue'],['deep56','boss']):

    mask = sints.get_act_mr3_crosslinked_mask(region)
    shape,wcs = mask.shape,mask.wcs

    w2 = np.mean(mask**2.)

    modlmap = mask.modlmap()

    szlab = "SMICA-nosz (PR3)"

    bin_edges = np.arange(20,3000,20)
    binner = stats.bin2D(modlmap,bin_edges)

    ppath = "/scratch/r/rbond/msyriac/data/planck/data/pr3/"

    imap = reproject.enmap_from_healpix(alm.copy(), shape, wcs, ncomp=1, unit=1, lmax=6000,
                                        rot="gal,equ", first=0, is_alm=True, return_alm=False) * mask

    nside = 2048
    pixwin = hp.pixwin(nside=nside,pol=False)
    pls = np.arange(len(pixwin))
    pwin = maps.interp(pls,pixwin)(modlmap)



    kmap = enmap.fft(imap,normalize='phys')/maps.gauss_beam(modlmap,fwhm) # *maps.gauss_beam(modlmap,dfwhm) /pwin #!!!!
    kmap[~np.isfinite(kmap)] = 0 
    p2d = (kmap*kmap.conj()).real / w2

    cents,p1d = binner.bin(p2d)
    pl.add(cents,p1d,color=col,label='Planck %s %s' % (szlab,region),ls="--")


#pl._ax.yaxis.set_minor_locator(AutoMinorLocator())
pl._ax.xaxis.set_minor_locator(AutoMinorLocator())
pl._ax.tick_params(axis='x',which='both', width=1)
pl._ax.tick_params(axis='y',which='both', width=1)
pl._ax.xaxis.grid(True, which='both',alpha=0.5)
pl._ax.yaxis.grid(True, which='both',alpha=0.5)

pl.done(os.environ['WORK']+"/fig_smica_nosz_check.pdf")
