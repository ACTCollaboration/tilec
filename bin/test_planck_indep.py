from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from szar import foregrounds as fgs
import soapack.interfaces as sints
from tilec import kspace,utils as tutils,ilc
from actsims import noise as simnoise
from enlib import bench


region = 'deep56'

mask = sints.get_act_mr3_crosslinked_mask(region)
modlmap = mask.modlmap()

w2 = np.mean(mask**2)
dm = sints.PlanckHybrid(region=mask)

qids = "p01,p02,p03,p04,p05,p06,p07,p08".split(',')

narrays = len(qids)
cbin_edges = np.arange(20,8000,20)
cbinner = stats.bin2D(modlmap,cbin_edges)

fpath = "/scratch/r/rbond/msyriac/data/depot/actsims/fg_res/fgfit_deep56"

for i in range(narrays):
    for j in range(i,narrays):
        pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='D',scalefn = lambda x: x**2.)

        qid1 = qids[i]
        qid2 = qids[j]
        array1 = sints.arrays(qid1,'freq')
        array2 = sints.arrays(qid2,'freq')

        splits1 = dm.get_splits([array1],ncomp=1,srcfree=True)[0,:,0]
        wins1 = dm.get_splits_ivar([array1])[0,:,0]

        splits2 = dm.get_splits([array2],ncomp=1,srcfree=True)[0,:,0]
        wins2 = dm.get_splits_ivar([array2])[0,:,0]


        fbeam1 = lambda x: tutils.get_kbeam(qid1,x,sanitize=True,planck_pixwin=True)
        fbeam2 = lambda x: tutils.get_kbeam(qid2,x,sanitize=True,planck_pixwin=True)

        ks1 = enmap.fft(splits1*mask,normalize='phys')/fbeam1(modlmap)
        ks2 = enmap.fft(splits2*mask,normalize='phys')/fbeam2(modlmap)

        p = ((ks1[0]*ks2[1].conj()).real + (ks1[1]*ks2[0].conj()).real)/2/w2


        cents,p1d = cbinner.bin(p)

        ls,s1d = np.loadtxt(f"{fpath}/s1d_{qid1}_{qid2}.txt",unpack=True)

        pl.add(ls,s1d,marker='o',ls='-',color="C0",markersize=3,label='s1d')
        #pl.add(ls,s1d/fbeam1(ls)/fbeam2(ls),marker='o',ls='-',color="C0",markersize=3,label='s1d')
        pl.add(cents,p1d,marker='d',ls='-',color="C1",markersize=3,label='cross')

        pl._ax.set_xlim(10,3500)
        #pl._ax.set_ylim(-1e2,1e5)
        pl._ax.set_ylim(10,1e8)
        pl.hline()
        pl.done(f"dcross_{qid1}_{qid2}.png")
