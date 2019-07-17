from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from tilec import utils as tutils

version = "v1.0.0_sint3"
qids  = "d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08".split(',')
#qids  = "p01,p02,p03,p04,p05,p06,p07,p08".split(',')

# version = "test_v1.0.0"
# qids  = "d56_01,p03,p04,p05".split(',')

cfunc = lambda x,y: "/scratch/r/rbond/msyriac/data/depot/tilec/%s/deep56/tilec_hybrid_covariance_%s_%s.hdf" % (version,x,y)
kfunc = lambda x: "/scratch/r/rbond/msyriac/data/depot/tilec/%s/deep56/kcoadd_%s.hdf" % (version,x)


mask = enmap.read_map("/scratch/r/rbond/msyriac/data/depot/tilec/%s/deep56/tilec_mask.fits"% version)
w2 = np.mean(mask**2)

narrays = len(qids)
for i in range(narrays):
    for j in range(i,narrays):
        qid1 = qids[i]
        qid2 = qids[j]
        if i!=j: continue # !!!

        p2d = enmap.read_map(cfunc(qid1,qid2))
        modlmap = p2d.modlmap()
        print(p2d[modlmap<80])
        shape,wcs = p2d.shape[-2:],p2d.wcs
        modlmap = enmap.modlmap(shape,wcs)
        bin_edges = np.append(np.append([0,20,40],np.arange(80,500,20)),np.arange(500,25000,20))
        #bin_edges = np.append(np.append([0,20,40],np.arange(80,300,20)),np.arange(300,25000,20))
        binner = stats.bin2D(modlmap,bin_edges)

        # b1 = tutils.get_kbeam(qid1,modlmap)
        # b2 = tutils.get_kbeam(qid2,modlmap)

        b1 = 1
        b2 = 1

        p2d = p2d/b1/b2


        k1 = enmap.read_map(kfunc(qid1))/b1
        k2 = enmap.read_map(kfunc(qid2))/b2 if qid1!=qid2 else k1
        kp2d = np.real(k1*k2.conj())/w2
        
        #pl = io.Plotter(xyscale='linlog',xlabel='$\\ell$',ylabel='$C_{\\ell}$')
        pl = io.Plotter(xyscale='loglog',xlabel='$\\ell$',ylabel='$C_{\\ell}$')
        cents,p1d = binner.bin(p2d)
        cents,p1d2 = binner.bin(kp2d)
        pl.add(cents,p1d)
        pl.add(cents,p1d2,ls='--')
        pl._ax.set_ylim(1e-5,1e5)
        pl.vline(x=80)
        pl.vline(x=500)
        pl.done("/scratch/r/rbond/msyriac/data/depot/tilec/plots/p1d_%s_%s.png" % (qid1,qid2))
        

