from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from soapack import interfaces as sints
from actsims import noise

froot = "/scratch/r/rbond/msyriac/data/scratch/tilec/test_lfi_v2_00_0000_deep56/"
kroot = "/scratch/r/rbond/msyriac/data/depot/tilec/test_lfi_v2_00_0000_deep56/"
droot = "/scratch/r/rbond/msyriac/data/depot/tilec/test_lfi_data_deep56/"
lroot = "/scratch/r/rbond/msyriac/data/depot/tilec/test_lfi_v3_00_0000_deep56/"
mask = sints.get_act_mr3_crosslinked_mask("deep56")
shape,wcs = mask.shape,mask.wcs
w2 = np.mean(mask**2.)
    
qids = 'p01,p02,p03'.split(',')

bin_edges = np.arange(20,6000,40)
modlmap = mask.modlmap()
binner = stats.bin2D(modlmap,bin_edges)

arraynames = {'p01':'030','p02':'044','p03':'070'}

dm = sints.PlanckHybrid(region=mask)
for qid in qids:


    split = enmap.read_map("%ssplit_%s.fits" % (froot,qid))
    #io.hplot(enmap.downgrade(split,4),"split_%s" % qid)

    arrayname = arraynames[qid]
    wts = dm.get_splits_ivar(arrayname)[0,:,0,...]

    coadd,_ = noise.get_coadd(split[:,0,...],wts,axis=0) * mask

    pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='C')

    kmap = enmap.fft(coadd,normalize='phys')
    p2d = np.real(kmap*kmap.conj())/w2
    cents,p1d = binner.bin(p2d)
    pl.add(cents,p1d,label='sim power')
    
    kcoadd = enmap.enmap(np.load("%skcoadd_%s.npy" % (kroot,qid)),wcs)
    p2d = np.real(kcoadd*kcoadd.conj())/w2
    cents,p1d = binner.bin(p2d)
    pl.add(cents,p1d,label='sim power 2')


    p2d = enmap.enmap(np.load("%stilec_hybrid_covariance_%s_%s.npy" % (kroot,qid,qid)),wcs)
    cents,p1d = binner.bin(p2d)
    pl.add(cents,p1d,label='sim cov')



    kcoadd = enmap.enmap(np.load("%skcoadd_%s.npy" % (droot,qid)),wcs)
    p2d = np.real(kcoadd*kcoadd.conj())/w2
    cents,p1d = binner.bin(p2d)
    pl.add(cents,p1d,label='data power',ls='--')


    p2d = enmap.enmap(np.load("%stilec_hybrid_covariance_%s_%s.npy" % (droot,qid,qid)),wcs)
    cents,p1d = binner.bin(p2d)
    pl.add(cents,p1d,label='data cov',ls='--')


    p2d = enmap.read_fits("/scratch/r/rbond/msyriac/dump/smoothed_beamed_%s_%s.fits" % (qid,qid))
    cents,p1d = binner.bin(p2d)
    pl.add(cents,p1d,label='sim S cov smoothed',ls='--')

    pl._ax.set_xlim(10,1000)
    pl._ax.set_ylim(1e-3,10)
    pl.vline(x=80)
    pl.done("lfi_power_comp_%s.png" % qid)


    
