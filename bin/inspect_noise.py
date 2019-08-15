from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap
import numpy as np
import os,sys

qids = "d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08".split(',')
theory = cosmology.default_theory()
ls = np.arange(2,3000)
cltt = theory.lCl('TT',ls)

pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='C')
pl.add(ls,cltt,color='k',lw=3)
for qid in qids:
    ells,n1d = np.loadtxt(os.environ['WORK']+'/n1d_%s.txt' % qid,unpack=True)
    n1d[ells<200] = 0
    pl.add(ells,n1d,ls='--' if 'p' in qid else '-')
    ells2,n1d2 = np.loadtxt(os.environ['WORK']+'/white_n1d_%s.txt' % qid,unpack=True)
    n1d2[ells2<200] = 0
    pl.add(ells2,n1d2,ls='--' if 'p' in qid else '-')
pl._ax.set_xlim(0,6000)
pl.done("noises.png")
