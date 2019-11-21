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

cversion = 'joint'
redo = False
ppath = "/scratch/r/rbond/msyriac/data/planck/data/pr2/"
fwhm = 10.
invbeam = lambda x: np.piecewise(x, [x<1,x>=1], [lambda y: y*0 , lambda y: 1./maps.gauss_beam(y,fwhm)])
#invbeam = lambda x: x*0 + 1

dfwhm = 0.0001

region_map = {'boss':'BOSS-N','deep56':'D56'}


pl = io.Plotter(xyscale='linlog',xlabel='$\\ell$',
                ylabel='$\\ell^2 C^{yy,\\rm{tot}}_{\\ell}/2\\pi~\\mathrm{(dimensionless)}$',
                scalefn = lambda x: x**2./2./np.pi,ftsize=16,labsize=14)
for col,region in zip(['red','blue'],['deep56','boss']):

    if redo:
        mask = sints.get_act_mr3_crosslinked_mask(region)
        shape,wcs = mask.shape,mask.wcs

        bfile = os.environ["WORK"] + "/data/depot/tilec/map_v1.0.0_rc_%s_%s/tilec_single_tile_%s_comptony_map_v1.0.0_rc_%s_beam.txt" % (cversion,region,region,cversion)
        yfile = os.environ["WORK"] + "/data/depot/tilec/map_v1.0.0_rc_%s_%s/tilec_single_tile_%s_comptony_map_v1.0.0_rc_%s.fits" % (cversion,region,region,cversion)
        w2 = np.mean(mask**2.)

        ls,bells = np.loadtxt(bfile,unpack=True)
        imap = enmap.read_map(yfile)
        modlmap = mask.modlmap()
        bin_edges = np.arange(20,6000,80)

        binner = stats.bin2D(modlmap,bin_edges)
        kmap = enmap.fft(imap,normalize='phys')/maps.interp(ls,bells)(modlmap)*maps.gauss_beam(modlmap,dfwhm)
        kmap[~np.isfinite(kmap)] = 0
        p2d = (kmap*kmap.conj()).real / w2

        cents,p1d = binner.bin(p2d)
        io.save_cols("ypower_tilec_%s.txt" % region, (cents,p1d))
    else:
        cents,p1d = np.loadtxt("ypower_tilec_%s.txt" % region, unpack=True)

    pl.add(cents,p1d,color=col,label='This work (%s)' % region_map[region])


    for ls,method in zip(['--',':'],['nilc','milca']):
        if method=='milca': continue

        if redo:

            bin_edges = np.arange(20,3000,80)
            binner = stats.bin2D(modlmap,bin_edges)

            alm = hp.read_alm(ppath + 'COM_CompMap_Compton-SZMap-%s-ymaps_2048_R2.00_alm.fits' % method)
            imap = reproject.enmap_from_healpix(alm, shape, wcs, ncomp=1, unit=1, lmax=6000,
                                                rot="gal,equ", first=0, is_alm=True, return_alm=False) * mask
            #io.hplot(enmap.downgrade(imap,4),"ymap_%s_%s" % (region,method),grid=True,color='gray')

            nside = 2048
            pixwin = hp.pixwin(nside=nside,pol=False)
            pls = np.arange(len(pixwin))
            pwin = maps.interp(pls,pixwin)(modlmap)


            kmap = enmap.fft(imap,normalize='phys')/maps.gauss_beam(modlmap,fwhm)*maps.gauss_beam(modlmap,dfwhm)/pwin
            kmap[~np.isfinite(kmap)] = 0 
            p2d = (kmap*kmap.conj()).real / w2

            cents,p1d = binner.bin(p2d)
            io.save_cols("ypower_planck_%s_%s.txt" % (region,method), (cents,p1d))
        else:
             cents,p1d  = np.loadtxt("ypower_planck_%s_%s.txt" % (region,method),unpack=True)
        pl.add(cents,p1d,color=col,ls=ls,label='Planck NILC %s' % region_map[region])


#pl._ax.yaxis.set_minor_locator(AutoMinorLocator())
pl._ax.xaxis.set_minor_locator(AutoMinorLocator())
pl._ax.tick_params(axis='x',which='both', width=1)
pl._ax.tick_params(axis='y',which='both', width=1)
pl._ax.xaxis.grid(True, which='both',alpha=0.3)
pl._ax.yaxis.grid(True, which='both',alpha=0.3)


pl.done("fig_ypower.pdf")
