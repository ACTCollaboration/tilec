from __future__ import print_function
import os,sys
import numpy as np
from soapack import interfaces as sints
from pixell import enmap,utils as putils,bunch
from tilec import tiling,kspace,ilc,pipeline,fg as tfg
from orphics import mpi, io,maps,catalogs,cosmology
from enlib import bench
from enlib.pointsrcs import sim_srcs
from tilec.utils import coadd,is_planck,apodize_zero,get_splits,get_splits_ivar,robust_ref,filter_div,get_kbeam,load_geometries,get_specs
from szar import foregrounds as szfg
b = bunch.Bunch


seed = 0

parent_qid = 'd56_01' # qid of array whose geometry will be used for the full map
ishape,iwcs = load_geometries([parent_qid])[parent_qid]
imap = enmap.zeros(ishape,iwcs)
imap2 = enmap.pad(imap,900)
shape,wcs = imap2.shape,imap2.wcs
nsplits = 2

"""
We will make 1 sim of:

1. Unlensed CMB
2. SZ Point sources
3. a Planck-143 like 2 split system
4. a Planck-100 like 2 split system
5. an S16 pa2 like 2 split system
6. an S16 pa3 like 2 split system with uncorrelated 90 and 150

"""



# Make the unlensed CMB realization
theory = cosmology.default_theory()
ells = np.arange(0,8000,1)
cltt = theory.lCl('TT',ells)
ps = cltt[None,None]
cmb = enmap.rand_map(shape, wcs, ps,seed=(0,seed))
modlmap = cmb.modlmap()
#io.plot_img(cmb,os.environ['WORK'] + '/tiling/cmbmap.png',lim=300)


# Make the SZ cluster realization
Nclusters = 800
amp_150_mean = 40
amp_150_sigma = 20
amps_150 = -np.abs(np.random.normal(amp_150_mean,amp_150_sigma,size=Nclusters))

def get_amps(freq):
    return amps_150 * szfg.ffunc(freq) / szfg.ffunc(150.)

arrays = b()
arrays.p143 = b() ;  arrays.p100 = b()  ; arrays.s16_01 = b()  ; arrays.s16_02 = b()  ; arrays.s16_03 = b()  
arrays.p143.freq = 143 ; arrays.p143.fwhm = 7. ; arrays.p143.rms = 30.
arrays.p100.freq = 100 ; arrays.p100.fwhm = 7.*(143./100.) ; arrays.p100.rms = 60.
arrays.s16_01.freq = 148 ; arrays.s16_01.fwhm = 1.4 ; arrays.s16_01.rms = 40.
arrays.s16_02.freq = 148 ; arrays.s16_02.fwhm = 1.4 ; arrays.s16_02.rms = 60.
arrays.s16_03.freq = 93 ; arrays.s16_03.fwhm = 1.4*(148./93.) ; arrays.s16_03.rms = 60.

np.random.seed((1,seed))
ras,decs = catalogs.random_catalog(shape,wcs,Nclusters,edge_avoid_deg=0.)

for qind,qid in enumerate(arrays.keys()):
    fwhm = arrays[qid].fwhm
    freq = arrays[qid].freq
    rms = arrays[qid].rms
    amps = get_amps(freq)

    srcs = np.stack((decs*putils.degree,ras*putils.degree,amps)).T
    szmap = sim_srcs(shape, wcs, srcs, beam=fwhm*putils.arcmin)

    kbeam = maps.gauss_beam(modlmap,fwhm)
    signal = maps.filter_map(cmb,kbeam) + szmap
    # io.hplot(signal,os.environ['WORK'] + '/tiling/signal_%s' % qid)

    for i in range(nsplits):
        noise = maps.white_noise(shape,wcs,rms,seed=(2,qind,seed,i))
        omap = signal + noise
        fname = os.environ['WORK'] + '/sim_tiling/%s_split_%d.fits' % (qid,i)
        enmap.write_map(fname,omap)
        io.hplot(omap,os.environ['WORK'] + '/tiling/total_%s_%d' % (qid,i))




