from __future__ import print_function
from orphics import maps,io,cosmology,catalogs
from pixell import enmap,reproject,powspec
import numpy as np
import os,sys
import ptfit
import soapack.interfaces as sints
from enlib import pointsrcs

"""
Caveats:
1. does not remove sources beyond abs(dec)=70 deg

"""


freq = sys.argv[1]
i = int(sys.argv[2])
cfile = sints.dconfig['planck_hybrid']['maps_path']+"/COM_PCCS_%s_R2.01.fits" % freq

cat = catalogs.load_fits(cfile,['RA','DEC','DETFLUX','DETFLUX_ERR','EXT_VAL'])
ras = np.deg2rad(cat['RA'])
decs = np.deg2rad(cat['DEC'])
fluxes = cat['DETFLUX']
sns = cat['DETFLUX']/cat['DETFLUX_ERR']
evals = cat['EXT_VAL']

arc = 40.
decmax = 70.
npix = int(arc/0.5)


assert np.all(fluxes>5.)
assert np.all(sns>5.)

print(len(sns[sns>5]))
print(len(sns[sns>10]))
print(len(sns[sns>10]))
print(len(sns[sns>20]))
#sys.exit()

ps = powspec.read_spectrum("input/cosmo2017_10K_acc3_scalCls.dat") # CHECK

dm = sints.PlanckHybrid()
pfwhm = dm.fwhms[freq]
nsplits = dm.get_nsplits(None,None,None)
if True:
    imap = dm.get_split(freq,i,ncomp=None,srcfree=True,pccs_sub=False)[0]
    fname = dm.get_split_fname(None,None,freq,i,srcfree=True,pccs_sub=False)
    oname = dm.get_split_fname(None,None,freq,i,srcfree=True,pccs_sub=True)
    assert "pccs" not in fname
    assert "pccs" in oname
    div = dm.get_split_ivar(freq,i,ncomp=None)[0]

    sras = []
    sdecs = []
    amps = []


    #ras,decs,iamps = np.loadtxt("/scratch/r/rbond/msyriac/data/planck/data/hybrid/planck_hybrid_545_2way_1_map_srcfree_pccs_sub_catalog.txt",unpack=True) # !!!


    for k,(ra,dec) in enumerate(zip(ras,decs)):
        if np.abs(np.rad2deg(dec))>decmax: continue
        #if iamps[k]>-500: continue # !!!
        # if sns[k]<20: continue
        # if k<300 or k>400: continue
        if evals[k]<3: continue

        stamp = reproject.cutout(imap, ra=ra, dec=dec, pad=1,  npix=npix)

        if stamp is None: continue
        divstamp = reproject.cutout(div, ra=ra, dec=dec, pad=1,  npix=npix)
        famp,cov,pfit = ptfit.ptsrc_fit(stamp,dec,ra,maps.sigma_from_fwhm(np.deg2rad(pfwhm/60.)),div=divstamp,ps=ps,beam=pfwhm,n2d=None)
        # model = pointsrcs.sim_srcs(stamp.shape, stamp.wcs, 
        #                            np.array((dec,ra,famp.reshape(-1)[0]))[None], 
        #                            maps.sigma_from_fwhm(np.deg2rad(pfwhm/60.)))
        # io.plot_img(np.log10(stamp),"stamp_%d.png" % k)
        # io.plot_img(divstamp,"divstamp_%d.png"  % k)
        # io.plot_img(model,"model_%d.png"  % k)
        # io.plot_img(stamp-model,"residual_%d.png"  % k)

        # if k==1: sys.exit()
        sdecs.append(dec)
        sras.append(ra)
        amps.append(famp.reshape(-1)[0])
        print(famp,sns[k])
        print("Done with source ", k+1, " / ",len(ras))


    srcs = np.stack((sdecs,sras,amps)).T 
    shape,wcs = imap.shape,imap.wcs
    model = pointsrcs.sim_srcs(shape[-2:], wcs, srcs, maps.sigma_from_fwhm(np.deg2rad(pfwhm/60.)))
    omap = imap - model
    mname = fname.replace('.fits','_pccs_sub_model.fits')
    cname = fname.replace('.fits','_pccs_sub_catalog.txt')
    io.save_cols(cname,(sras,sdecs,amps))
    enmap.write_fits(mname,model[None])
    enmap.write_fits(oname,omap[None])
