from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
#plt.rcParams["mathtext.fontset"] = "dejavuserif"
from orphics import maps,io,cosmology,mpi,stats
from pixell import enmap,curvedsky,utils as putils
import numpy as np
import os,sys,shutil
from actsims.util import seed_tracker
from soapack import interfaces as sints
from enlib import bench
from tilec import pipeline,utils as tutils
import healpy as hp
from szar import foregrounds as fgs
from datetime import datetime
from tilec.pipeline import get_input
from actsims.util import seed_tracker



region = 'deep56'
tdir = "/scratch/r/rbond/msyriac/data/depot/tilec/"
dcomb = 'joint'

if False:
    qids = "d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08".split(',')
    #qids = "d56_01,d56_02".split(',')

    pows = {}

    seeds = [12,11]

    for seed in seeds:
        sim_idx = seed
        set_idx = 0
        fgres_seed = seed_tracker.get_fg_seed(set_idx, sim_idx, 'srcfree')
        jsim = pipeline.JointSim(qids,fg_res_version="fgfit_deep56",ellmax=8101,bandpassed=True,no_act_color_correction=False,ccor_exp=-1)
        alms = curvedsky.rand_alm_healpy(jsim.cfgres, seed = fgres_seed)
        narrays = alms.shape[0]

        for i in range(narrays):
            for j in range(i,narrays):
                qid1 = qids[i]
                qid2 = qids[j]
                cls = hp.alm2cl(alms[i],alms[j])
                ls = np.arange(cls.size)
                pows[qid1+qid2+str(seed)] = cls.copy()

    for i in range(narrays):
        for j in range(i,narrays):
            qid1 = qids[i]
            qid2 = qids[j]
            pl = io.Plotter(xyscale='linlog',scalefn = lambda x: x**2./2./np.pi, xlabel='l',ylabel='D')
            for seed in seeds:
                cls = pows[qid1+qid2+str(seed)]
                pl.add(ls,cls,label=str(seed))
            pl.done(f"fgres_simpow_{qid1}_{qid2}.png")



seeds = [11,12]

p = lambda x: (x*x.conj()).real

bin_edges = np.arange(20,6000,20)


versions = ['test_sim_galtest_nofg','test_sim_galtest_withfg_fgfit','test_sim_galtest_withfg_test']
for version in versions:
    pl = io.Plotter(xyscale='linlog',scalefn = lambda x: x**2./2./np.pi,xlabel='l',ylabel='D')
    for seed in seeds:
        csfile = tutils.get_generic_fname(tdir,region,'cmb',deproject=None,data_comb=dcomb,version=version,sim_index=seed)
        imap = enmap.read_map(csfile)
        modlmap = imap.modlmap()
        k = enmap.fft(imap,normalize='phys')
        p2d = p(k)
        binner = stats.bin2D(modlmap,bin_edges)
        cents,p1d = binner.bin(p2d)
        pl.add(cents,p1d,lw=1,alpha=0.8,label=f'{seed}')
    pl._ax.set_ylim(10,3e5)
    pl.done("cpowall_%s.png" % version)


# #This snippet discovered that sim_index=12 is the first instance of break-down
# nsims = 13

# p = lambda x: (x*x.conj()).real

# bin_edges = np.arange(20,6000,20)

# pl = io.Plotter(xyscale='linlog',scalefn = lambda x: x**2./2./np.pi,xlabel='l',ylabel='D')

# for i in range(nsims):
#     csfile = tutils.get_generic_fname(tdir,region,'cmb',deproject=None,data_comb=dcomb,version="sim_baseline",sim_index=i)
#     imap = enmap.read_map(csfile)
#     modlmap = imap.modlmap()
#     k = enmap.fft(imap,normalize='phys')
#     p2d = p(k)
#     binner = stats.bin2D(modlmap,bin_edges)
#     cents,p1d = binner.bin(p2d)
#     pl.add(cents,p1d,lw=1,alpha=0.8)
#     if np.any(p1d*cents**2>1e5): print(i)
# pl.done("cpow.png")
