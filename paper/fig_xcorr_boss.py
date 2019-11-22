from __future__ import print_function
from orphics import maps,io,cosmology,catalogs,stats
from pixell import enmap,curvedsky as cs
import numpy as np
import os,sys
from tilec import utils as tutils
import healpy as hp

nside = 2048
tdir = '/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919'
lmax = 5000


# data_files = ['galaxy_DR12v5_CMASS_North.fits','galaxy_DR12v5_CMASS_South.fits']
# rfiles = ['random0_DR12v5_CMASS_North.fits','random1_DR12v5_CMASS_North.fits',
#           'random0_DR12v5_CMASS_South.fits','random1_DR12v5_CMASS_South.fits']

# cmapper = catalogs.BOSSMapper(['/scratch/r/rbond/msyriac/data/boss/boss_dr12/' + x for x in data_files],random_files=['/scratch/r/rbond/msyriac/data/boss/boss_dr12/' + x for x in rfiles],rand_sigma_arcmin=2.,rand_threshold=1e-3,nside=nside,verbose=True,hp_coords="equatorial")

# hp.write_map(os.environ['WORK'] +'/delta.fits',cmapper.get_delta())
# hp.write_map(os.environ['WORK'] +'/mask.fits',cmapper.mask)

hdelta = hp.read_map(os.environ['WORK'] +'/delta.fits')
mask = hp.read_map(os.environ['WORK'] +'/mask.fits')

# io.mollview(hdelta,'hdelta.png')
# io.mollview(mask,'hmask.png')

#sys.exit()
pl = io.Plotter(xlabel='l',ylabel='LC_L',scalefn = lambda x: x,xyscale='linlin')
      
def get_beam(bfile,ells):
    ls,bells = np.loadtxt(bfile,unpack=True)
    return maps.interp(ls,bells)(ells)

for col,region in zip(['red','blue'],['boss','deep56']):



    ymap = enmap.read_map(tutils.get_generic_fname(tdir,region,"tsz",None,"joint"))
    cmap = enmap.read_map(tutils.get_generic_fname(tdir,region,"tsz",'cmb',"joint"))
    dmap = enmap.read_map(tutils.get_generic_fname(tdir,region,"tsz",'cib',"joint"))

    ybfile = tutils.get_generic_fname(tdir,region,"tsz",None,"joint",beam=True)
    cbfile = tutils.get_generic_fname(tdir,region,"tsz",'cmb',"joint",beam=True)
    dbfile = tutils.get_generic_fname(tdir,region,"tsz",'cib',"joint",beam=True)
    
    imask = enmap.read_map(tutils.get_generic_fname(tdir,region,"tsz",None,"joint",mask=True))
    dmask = hp.alm2map(cs.map2alm(imask,lmax=lmax).astype(np.complex128),nside)

    dmask[dmask<0.5] = 0
    dmask[dmask>0.5] = 1
    jmask = dmask*mask
    io.mollview(jmask,os.environ['WORK'] + '/jmask_%s.png' % region)

    fsky = jmask.sum()/jmask.size
    print(fsky * 41252.)
    delta  = hdelta * jmask
    galm = hp.map2alm(delta,lmax=lmax)

    for bfile,imap in zip([ybfile,cbfile,dbfile],[ymap,cmap,dmap]):
        yalm = hp.map2alm(hp.alm2map(cs.map2alm(imap,lmax=lmax).astype(np.complex128),nside=nside)*jmask,lmax=lmax)
        cls = hp.alm2cl(galm,yalm)
        assert np.all(np.isfinite(cls))
        ells = np.arange(len(cls))
        lbeam = get_beam(bfile,ells)
        bin_edges = np.arange(320,5000,400)
        binner = stats.bin1D(bin_edges)
        cls = cls / lbeam
        cents,bcls = binner.binned(ells,cls)
        bcls[~np.isfinite(bcls)] = 0
        pl.add(cents,bcls/fsky,marker="o",color=col)








pl.hline(y=0)
pl.done(os.environ['WORK'] + "/fig_xcorr_cls.png")
