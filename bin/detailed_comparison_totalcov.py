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
narrays = len(qids)

Ny,Nx = modlmap.shape

redo = True

if redo:
    for seed in seeds:

        usig = np.zeros((Ny,Nx,narrays,narrays))
        #ssig = np.zeros((Ny,Nx,narrays,narrays))

        for i in range(narrays):
            for j in range(i,narrays):
                qid1 = qids[i]
                qid2 = qids[j]


                if True:
                    fname = f"/scratch/r/rbond/msyriac/data/scratch/tilec/test_sim_galtest_nofg_{version}_00_00{seed}_deep56/scovs_{i}_{j}.npy"
                    scov = enmap.enmap(np.load(fname),modlmap.wcs)

                    fname = f"/scratch/r/rbond/msyriac/data/scratch/tilec/test_sim_galtest_nofg_{version}_00_00{seed}_deep56/unsmoothed_noise_cov_{i}_{j}.npy"
                    try:
                        ncov = enmap.enmap(np.load(fname),modlmap.wcs)
                    except:
                        ncov = 0

                    ccov = scov + ncov
                else:
                    fname = f"/scratch/r/rbond/msyriac/data/scratch/tilec/test_sim_galtest_nofg_{version}_00_00{seed}_deep56/coadd_cov_{i}_{j}.npy"
                    ccov = enmap.enmap(np.load(fname),modlmap.wcs)


                usig[...,i,j] = usig[...,j,i] = ccov.copy()
                print(i,j)

    print("eigs...")
    emap = modlmap*0
    # v = np.linalg.eigh(usig[modlmap<6000,...][500,...])[0]
    # print(v)
    # print(v.shape)
    emap[modlmap<6000] = np.linalg.eigh(usig[modlmap<6000,...])[0][...,0]
    #emap = np.linalg.eigh(usig)[0][:,:,0]
    np.save("emap_tot",emap)
else:
    emap = np.load("emap_tot.npy")
emap[emap<-1e-10] = np.nan
io.power_crop(np.fft.fftshift(emap),300,"eigmapdet.png",ftrans=False,lim=[-0.01,0.01])
