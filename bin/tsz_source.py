from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap,pointsrcs,reproject
import numpy as np
import os,sys
from tilec import fg as tfg,utils as tutils
from szar import foregrounds as sfg
from soapack import interfaces as sints

fg = 'tSZ'

def ffunc(nu,comp):
    if comp=='tSZ' or comp=='CIB':
        return tfg.get_mix(nu,comp)[0]
    elif comp=='rad':
        return sfg.rad_ps_nu(nu)
        

nus = np.geomspace(25,600,1000)

srad = sfg.rad_ps_nu(nus)

tsz = tfg.get_mix(nus,'tSZ')/1e6
tdust = tfg.get_mix(nus,'CIB')/tfg.get_mix(150,'CIB')

sdust = sfg.cib_nu(nus)/sfg.cib_nu(150)

ssz = sfg.ffunc(nus)*2.725

pl = io.Plotter(xyscale='loglog',xlabel='$\\nu$ (GHz)',ylabel='$f(\\nu)$',figsize=(6,3))
pl.add(nus,srad,label='Synchrotron',color='C0')

nnus = nus[tsz<=0]
ntsz = tsz[tsz<=0]

pnus = nus[tsz>0]
ptsz = tsz[tsz>0]

pl.add(nnus,-ntsz,label='tSZ',ls='--',color='C1')
pl.add(pnus,ptsz,color='C1')
pl.add(nus,tdust,label='Dust',color='C2')
pl.done("fg_SED.png")

sys.exit()

qids = "d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08".split(',')
beams = [1.4,1.4,1.4,1.4,2.2,1.4,32.408,27.100,13.315,9.69,7.30,5.02,4.94,4.83]
region = 'deep56'

mask = sints.get_act_mr3_crosslinked_mask(region)
shape,wcs = mask.shape,mask.wcs

aspecs = tutils.ASpecs().get_specs
for i,qid in enumerate(qids):

    lmin,lmax,hybrid,radial,friend,cfreq,fgroup,wrfit = aspecs(qid)
    nu = cfreq
    dec = 0
    ra = 10
    srcs = np.array([[np.deg2rad(dec),np.deg2rad(ra),ffunc(nu,fg)]])
    beam = maps.sigma_from_fwhm(np.deg2rad(beams[i]/60)) 

    imap = pointsrcs.sim_srcs(shape, wcs, srcs, beam, omap=None, dtype=None, nsigma=5, rmax=None, smul=1,
                              return_padded=False, pixwin=False, op=np.add, wrap="auto", verbose=False, cache=None)

    #cut = reproject.cutout(imap,ra=np.deg2rad(ra),dec=np.deg2rad(dec),npix=60)
    kmap = enmap.fft(imap,normalize='phys')
    np.save("/scratch/r/rbond/msyriac/data/depot/tilec/%s_src_test_v1.0.0_rc_deep56/kcoadd_%s.npy" % (fg.lower(),qid), kmap)
    print(qid)
    #io.plot_img(cut)
