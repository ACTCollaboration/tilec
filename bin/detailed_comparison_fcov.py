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

#version = "noLFI_nohigh_test"
version = "noLFI_nohigh_test_int0"

mask = sints.get_act_mr3_crosslinked_mask(region)
modlmap = mask.modlmap()

# cov = 1./modlmap**0.5
# cov[~np.isfinite(cov)] = 0
# bcov = covtools.signal_average(cov,bin_width=160,kind=0,lmin=20,dlspace=True)
# bcov2 = covtools.signal_average(bcov,bin_width=160,kind=0,lmin=20,dlspace=True)
# #assert np.all(np.isclose(bcov,bcov2))
# io.power_crop(bcov,150,"binmatdet.png",ftrans=True,lim=[-2,-1])
# io.power_crop(bcov2,150,"binmatdet2.png",ftrans=True,lim=[-2,-1])


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
        c = 0
        for i in range(narrays):
            for j in range(i,narrays):
                qid1 = qids[i]
                qid2 = qids[j]


                fname = f"/scratch/r/rbond/msyriac/data/depot/tilec/test_sim_galtest_nofg_{version}_00_00{seed}_deep56/tilec_hybrid_covariance_{qid1}_{qid2}.npy"
                ccov = enmap.enmap(np.load(fname),modlmap.wcs)


                #cents,c1d,_ = covtools.signal_average(ccov,bin_width=160,kind=0,lmin=20,dlspace=True,return_bins=True)
                cents,c1d,_ = covtools.signal_average(ccov,bin_width=160,kind=0,lmin=20,dlspace=False,return_bins=True)
                if c==0:
                    icents = cents[cents<2000]
                    nbins = len(icents)
                    bsig = np.zeros((nbins,narrays,narrays))

                bsig[...,i,j] = bsig[...,j,i] = c1d[cents<2000].copy()

                usig[...,i,j] = usig[...,j,i] = ccov.copy()
                print(i,j)
                c = c + 1

    print("eigs...")
    # emap = modlmap*0
    # emap[modlmap<6000] = np.linalg.eigh(usig[modlmap<6000,...])[0][...,0]
    # np.save("emap_tot",emap)
    
    eigs = np.linalg.eigh(bsig)[0]
    print(eigs.shape)
    

else:
    # emap = np.load("emap_tot.npy")
    pass

pl = io.Plotter(xlabel='l',ylabel='e')
for i in range(eigs.shape[1]):
    E = eigs[:,i]
    print(E)
    pl.add(icents,np.log(np.maximum(E,1e-4)),marker="o",ls="none")
pl.done("binned_eigs.png")
    

# selem = emap[np.logical_and(modlmap>900,modlmap<1400)]
# print(selem.min())
# emap[emap<-1e-14] = np.nan
# #io.power_crop(np.fft.fftshift(np.log10(np.abs(emap))),150,"eigmapdet.png",ftrans=False)#,lim=[-0.01e-2,0.01e-2])
# io.power_crop(np.fft.fftshift(emap),150,"eigmapdet.png",ftrans=False,lim=[-0.01e-2,0.01e-2])

