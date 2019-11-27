from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from soapack import interfaces as sints
from tilec import utils as tutils,covtools
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


region = 'deep56'

version = "noLFI_nohigh_test"

mask = sints.get_act_mr3_crosslinked_mask(region)
modlmap = mask.modlmap()
lmap = mask.lmap()
bin_edges = np.arange(20,6000,80)
binner = stats.bin2D(modlmap,bin_edges)
def pow(x,y=None):
    k = enmap.fft(x,normalize='phys')
    ky = enmap.fft(y,normalize='phys') if y is not None else k
    p = (k*ky.conj()).real
    cents,p1d = binner.bin(p)
    return p,cents,p1d

seeds = [12]#,13]

qids = "d56_04,d56_05,d56_06,p04,p05,p06".split(',')
fs = [150,90,150,90,150,217]
fdict = {}
for qid,f in zip(qids,fs):
    fdict[qid]= f
narrays = len(qids)

Ny,Nx = modlmap.shape

for seed in seeds:

    for i in range(narrays):
        for j in range(i,narrays):

            

            pl = io.Plotter('Dell')
            qid1 = qids[i]
            qid2 = qids[j]



            beam1 = tutils.get_kbeam(qid1,modlmap,sanitize=False,planck_pixwin=True)
            beam2 = tutils.get_kbeam(qid2,modlmap,sanitize=False,planck_pixwin=True)
            fname = f"/scratch/r/rbond/msyriac/data/scratch/tilec/test_sim_galtest_nofg_{version}_00_00{seed}_deep56/scovs_{i}_{j}.npy"

            # if qid1=='p04' and qid2=='p04':
            #     pass
            # else:
            #     continue

            scov = enmap.enmap(np.load(fname),modlmap.wcs)/beam1/beam2

            # io.power_crop(scov,500,"detp04p2d.png",ftrans=True)
            # continue

            cents,s1d = binner.bin(scov)
            pl.add(cents,s1d,label='single array')

            fname = f"/scratch/r/rbond/msyriac/data/scratch/tilec/test_sim_galtest_nofg_noLFI_nohigh_test_00_0012_deep56/fscovs_{i}_{j}.npy"
            scov = enmap.enmap(np.load(fname),modlmap.wcs)
            cents,f1d1 = binner.bin(scov)
            pl.add(cents,f1d1,label=f'coadded')

            pl._ax.set_xlim(0,4000)
            #pl._ax.set_ylim(1,1e4)
            
            pl.done(f"c1dcoadd_det_{seed}_{qid1}_{qid2}.png")
