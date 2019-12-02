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
from tilec import utils as tutils
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
narrays = len(qids)

for comp in ['cmb']:
    for i in range(narrays):
        qid = qids[i]
        for seed in seeds:

            fname = f"/scratch/r/rbond/msyriac/data/scratch/tilec/test_sim_galtest_nofg_{version}_00_00{seed}_deep56/dncovs_{i}_{i}.npy"
            scov = enmap.enmap(np.load(fname),modlmap.wcs)
            io.power_crop(scov,600,f"dn2d_det_{qid}_{seed}.png")

