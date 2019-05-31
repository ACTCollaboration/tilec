from __future__ import print_function
from orphics import maps,io,cosmology,catalogs
from pixell import enmap,reproject,powspec,utils
import numpy as np
import os,sys
import ptfit
import soapack.interfaces as sints
from enlib import pointsrcs


freq = sys.argv[1]
i = int(sys.argv[2])
cfile = sints.dconfig['planck_hybrid']['maps_path']+"/COM_PCCS_%s_R2.01.fits" % freq


ffactors = {'545': 0.01723080316,
            '857': 0.44089766765}

beam_area = {"545":26.44,"857":24.37} # arcmin^2

# amps1 = 1*1e-3/utils.flux_factor(beam_area=beam_area[freq]*(np.pi/180./60.)**2.,freq=float(freq)*1e9)*1e6
# amps2 = 1*1e-3*(ffactors[freq]/1e6)/(beam_area[freq]*(np.pi/180./60.)**2.)*1e6
# print(amps1,amps2)
# sys.exit()


cat = catalogs.load_fits(cfile,['RA','DEC','DETFLUX','DETFLUX_ERR','EXT_VAL'])
ras = np.deg2rad(cat['RA'])
decs = np.deg2rad(cat['DEC'])
fluxes = cat['DETFLUX']
sns = cat['DETFLUX']/cat['DETFLUX_ERR']
evals = cat['EXT_VAL']

assert np.all(fluxes>5.)
assert np.all(sns>5.)


ps = powspec.read_spectrum("input/cosmo2017_10K_acc3_scalCls.dat") # CHECK

dm = sints.PlanckHybrid()
pfwhm = dm.fwhms[freq]
nsplits = dm.get_nsplits(None,None,None)

imap = dm.get_split(freq,i,ncomp=None,srcfree=False)[0]
fname = dm.get_split_fname(None,None,freq,i,srcfree=False)
oname = dm.get_split_fname(None,None,freq,i,srcfree=True)
assert "srcfree" not in fname
assert "srcfree" in oname

#amps = fluxes*1e-3/utils.flux_factor(beam_area=beam_area[freq]*(np.pi/180./60.)**2.,freq=float(eff_freqs[freq])*1e9)*1e6
amps = fluxes*1e-3*(ffactors[freq]/1e6)/(beam_area[freq]*(np.pi/180./60.)**2.)*1e6
sras = ras
sdecs = decs

srcs = np.stack((sdecs,sras,amps)).T 
shape,wcs = imap.shape,imap.wcs
model = pointsrcs.sim_srcs(shape[-2:], wcs, srcs, maps.sigma_from_fwhm(np.deg2rad(pfwhm/60.)))
omap = imap - model
mname = fname.replace('.fits','_pccs_sub_model.fits')
cname = fname.replace('.fits','_pccs_sub_catalog.txt')
io.save_cols(cname,(sras,sdecs,amps))
enmap.write_fits(mname,model[None])
enmap.write_fits(oname,omap[None])
