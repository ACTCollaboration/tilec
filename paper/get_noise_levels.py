from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap
import numpy as np
import os,sys
import soapack.interfaces as sints


"""
Referee has requested noise levels in each array.
So we use the noise estimated from auto - cross
for the foreground fitting spectra.
"""

version = 'fgfit_v2'

arrays = {'boss':'boss_01,boss_02,boss_03,boss_04,p01,p02,p03,p04,p05,p06,p07,p08',
          'deep56':'d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08'}


planck = {}
for qid in 'p01,p02,p03,p04,p05,p06,p07,p08'.split(','):
    planck[qid] = 0

for region in ['boss','deep56']:
    spath = sints.dconfig['actsims']['fg_res_path'] + "/"+ version + "_" + region +  "/"
    qids = arrays[region].split(',')
    for qid in qids:

        ncents,n1d = np.loadtxt("%sn1d_%s_%s.txt" % (spath,qid,qid), unpack=True)
        if qid in ['p01','p02']:
            lmin = 150
            lmax = 290
        elif qid=='p03':
            lmin = 1000
            lmax = 1990
        else:
            lmin = 5000
            lmax = 5500
        sel = np.logical_and(ncents>lmin,ncents<lmax)
        noise = np.sqrt(n1d[sel].mean()) * 180.*60./np.pi
        print(region, qid, noise)

        if 'p' in qid:
            planck[qid] = planck[qid] + noise/2.


for qid in 'p01,p02,p03,p04,p05,p06,p07,p08'.split(','):
    print(qid,planck[qid])
