from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap
import numpy as np
import os,sys
from tilec import utils as tutils

qids = 'boss_01,boss_02,boss_03,boss_04,d56_01,d56_02,d56_03,d56_04,d56_05,d56_06'.split(',')

ells = np.arange(2,8000,1)

for qid in qids:

    old = tutils.get_kbeam(qid,ells,sanitize=False,version='190220',planck_pixwin=False)
    new = tutils.get_kbeam(qid,ells,sanitize=False,version='190809',planck_pixwin=False)

    pl = io.Plotter(xlabel='l',ylabel='r')
    pl.add(ells,(new-old)/old,label=qid)
    pl.done("beamcomp_%s.png" % qid)
