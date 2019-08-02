from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from tilec import utils as tutils

#version = "v1.1.0_rc_deep56"
version = "test_v1.1.0_rc_deep56"
qids  = "d56_01,d56_02,p06".split(',')
#qids  = "d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08".split(',')
#qids  = "p01,p02,p03,p04,p05,p06,p07,p08".split(',')
#qids = "d56_05,d56_06,p01,p04,p05,p06".split(',')

# version = "test_v1.0.0"
# qids  = "d56_01,p03,p04,p05".split(',')

cfunc = lambda x,y: "/scratch/r/rbond/msyriac/data/depot/tilec/%s/tilec_hybrid_covariance_%s_%s.npy" % (version,x,y)
kfunc = lambda x: "/scratch/r/rbond/msyriac/data/depot/tilec/%s/kcoadd_%s.npy" % (version,x)


mask = enmap.read_map("/scratch/r/rbond/msyriac/data/depot/tilec/%s/tilec_mask.fits"% version)
w2 = np.mean(mask**2)
shape,wcs = mask.shape,mask.wcs

narrays = len(qids)
for i in range(narrays):
    for j in range(i,narrays):
        qid1 = qids[i]
        qid2 = qids[j]
        #if i!=j: continue # !!!

        p2d = enmap.enmap(np.load(cfunc(qid1,qid2)),wcs)
        modlmap = p2d.modlmap()
        print(p2d[modlmap<80])
        modlmap = enmap.modlmap(shape,wcs)
        bin_edges = np.append(np.append([0,20,40],np.arange(80,500,20)),np.arange(500,25000,20))
        #bin_edges = np.append(np.append([0,20,40],np.arange(80,300,20)),np.arange(300,25000,20))
        binner = stats.bin2D(modlmap,bin_edges)

        # b1 = tutils.get_kbeam(qid1,modlmap)
        # b2 = tutils.get_kbeam(qid2,modlmap)

        b1 = 1
        b2 = 1

        p2d = p2d/b1/b2


        k1 = enmap.enmap(np.load(kfunc(qid1))/b1,wcs)
        k2 = enmap.enmap(np.load(kfunc(qid2))/b2,wcs) if qid1!=qid2 else k1
        kp2d = np.real(k1*k2.conj())/w2
        
        #pl = io.Plotter(xyscale='linlog',xlabel='$\\ell$',ylabel='$C_{\\ell}$')
        pl = io.Plotter(xyscale='loglog',xlabel='$\\ell$',ylabel='$C_{\\ell}$')
        cents,p1d = binner.bin(p2d)
        cents,p1d2 = binner.bin(kp2d)
        pl.add(cents,p1d)
        pl.add(cents,p1d2,ls='--')
        pl._ax.set_ylim(1e-7,1e5)
        pl.vline(x=80)
        pl.vline(x=500)
        pl.done("/scratch/r/rbond/msyriac/data/depot/tilec/plots/p1d_%s_%s.png" % (qid1,qid2))
        

