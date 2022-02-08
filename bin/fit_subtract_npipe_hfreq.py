#%%

"""
The calibration in 545 GHz has changed
with NPIPE and a new PCCS catalog is
not publicly available.

We will instead use the old PCCS catalog
and fit the fluxes in the new maps, 
then subtract from the new maps.
"""

#%load_ext autoreload
#%autoreload 2

from pixell import enmap,curvedsky as cs, utils as u,reproject
import numpy as np
from orphics import maps,io,catalogs,cosmology
import os,sys
from actsims import noise as anoise
import ptfit
import soapack.interfaces as sints
from pixell import pointsrcs
from enlib import bench
# %%

def fit_beam(imap, coords, fwhm,ps,noise,r=5*u.arcmin, verbose=False,
		):
    """Given an enmap [...,ny,nx] and a set of coords [n,{dec,ra}], extract a set
	of cutouts [n,...,thumby,thumbx] centered on each set of
	coordinates. Fit a beam to each cutout, and return the fluxes."""
    
    print("Getting thumbnails...")
    thumbs = reproject.thumbnails(imap,coords,r=r,proj='tan',pixwin=False)

    shape,wcs = thumbs.shape[-2:],thumbs.wcs
    ivar = maps.ivar(shape,wcs,noise)

    print("Initializing fitter...")
    pfitter = ptfit.Pfit(shape,wcs,rbeam=None,div=ivar,ps=ps,beam=fwhm,n2d=None,invert=False)

    nthumbs = coords.shape[0]
    rbeam = maps.sigma_from_fwhm(np.deg2rad(fwhm/60.))
    fluxes = []
    print("Fitting thumbnails...")
    for i in range(nthumbs):
        pflux,cov,fit,_ = pfitter.fit(thumbs[i],dec=0,ra=0,rbeam=rbeam)
        fluxes.append(pflux[0])
        print(f"Source {i+1} / {nthumbs} done.")
    fluxes = np.asarray(fluxes)
    
    out = np.zeros((nthumbs,3))
    out[:,0] = coords[:,0]
    out[:,1] = coords[:,1]
    out[:,2] = fluxes
    print("Making source map")
    smap = pointsrcs.sim_srcs(imap.shape[-2:], imap.wcs, out, rbeam)

    return smap
    
#%%
splitnum = int(sys.argv[1])
freq = 545

cfile = sints.dconfig['planck_hybrid']['maps_path']+"/COM_PCCS_%s_R2.01.fits" % freq

print("Loading catalog...")

cat = catalogs.load_fits(cfile,['RA','DEC','DETFLUX','DETFLUX_ERR','EXT_VAL'])
ras = np.deg2rad(cat['RA'])
decs = np.deg2rad(cat['DEC'])


#%%
dfact = 4

def hdowngrade(imap,shape,wcs,lmax):
    return(cs.alm2map(cs.map2alm(imap,lmax=lmax),enmap.empty(shape,wcs,dtype=imap.dtype)))


i = splitnum
fname = f"/home/r/rbond/sigurdkn/project/actpol/planck/npipe/car_equ/planck_npipe_{freq}_split{i+1}_map.fits"

hlmax = 12000
nside = 2048
llmax = 3*nside

print("Loading map...")
oimap = enmap.read_map(fname,sel=np.s_[0,...])
oshape,owcs = enmap.fullsky_geometry(res=2.0 * u.arcmin)
print("Downgrading map...")
imap = hdowngrade(oimap,oshape,owcs,lmax=hlmax)
    



# %%

ells = np.arange(9000)
theory = cosmology.default_theory()
cltt = theory.lCl('TT',ells)

smap = fit_beam(imap,np.asarray((decs,ras)).T,fwhm=4.83,ps=cltt,noise=818.2,r=15*u.arcmin)

print("Upgrading map...")
omap = hdowngrade(smap,oimap.shape,oimap.wcs,lmax=llmax)
print("Writing map...")
enmap.write_map(f'/scratch/r/rbond/msyriac/data/planck/npipe/srcsub/npipe_{freq}_{splitnum}_srcmap.fits',omap)

# %%
