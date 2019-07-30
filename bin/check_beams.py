from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap
import numpy as np
import os,sys
from tilec import utils as tutils



qids = ['d56_%s' % str(i+1).zfill(2) for i in range(6)]

ells = np.arange(2,35000)
pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='B')
for i,qid in enumerate(qids):
    lbeam = tutils.get_kbeam(qid,ells)
    pl.add(ells,np.abs(lbeam),label=qid,color="C%d" % i,ls="-",alpha=0.7)
    pl.add(ells,tutils.sanitize_beam(ells,lbeam),ls="--",color="C%d" % i)
pl._ax.set_ylim(1e-8,2)
pl.done("lbeam.png")

