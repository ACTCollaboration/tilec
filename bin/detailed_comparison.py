from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
import soapack.interfaces as sints
from tilec import kspace,utils as tutils
from actsims import noise as simnoise
from enlib import bench

region = 'deep56'
version = "test_sim_galtest_final"
#version = "noLFI_nohigh_test_int0"
#version = "noLFI_yesLFI_test"


mask = sints.get_act_mr3_crosslinked_mask(region)
modlmap = mask.modlmap()
lmap = mask.lmap()
bin_edges = np.arange(20,6000,20)
binner = stats.bin2D(modlmap,bin_edges)
def pow(x,y=None):
    k = enmap.fft(x,normalize='phys')
    ky = enmap.fft(y,normalize='phys') if y is not None else k
    p = (k*ky.conj()).real
    cents,p1d = binner.bin(p)
    return p,cents,p1d

droot = "/scratch/r/rbond/msyriac/data/depot/tilec/"
sroot = "/scratch/r/rbond/msyriac/data/scratch/tilec/"

#seeds = [11,12]
seeds = [12,13]

pl = io.Plotter('Dell')
for seed in seeds:
    #fname = f"{droot}map_joint_test_sim_galtest_nofg_00_00{seed}_deep56/tilec_single_tile_deep56_cmb_map_joint_test_sim_galtest_nofg_00_00{seed}.fits"
    fname = f"{droot}map_joint_{version}_00_00{seed}_deep56/tilec_single_tile_deep56_cmb_map_joint_{version}_00_00{seed}.fits"
    imap = enmap.read_map(fname)
    p2d,cents,p1d = pow(imap)
    lp2d = np.log10(p2d)
    sel = np.logical_and(modlmap<1300,modlmap>1100)
    vals = lp2d[sel]
    # print(vals)
    # print(vals.mean(),vals.max(),vals.min(),vals.std())
    hist,edges = np.histogram(vals,bins=np.linspace(-15,5,300))
    hcents = (edges[1:]+edges[:-1])/2.
    pl2 = io.Plotter(xlabel='v',ylabel='N')
    pl2.add(hcents,hist)
    pl2.done(f"detvalhist{seed}.png")


    pl.add(cents,p1d,label=f"{seed}")
    
    llim = 0
    io.power_crop(p2d,150,f"det2d{seed}comp.png",lim=[-10,1])
    badinds = np.argwhere(np.logical_and(np.log10(p2d)>llim,modlmap>1000))
    print(modlmap[np.logical_and(np.log10(p2d)>llim,modlmap>1000)])
    #sys.exit()
    badinds = np.argwhere(np.logical_and(np.log10(p2d)>llim,modlmap>1000))
    if len(badinds)>0:
        ind = 0
        print(badinds[ind])
        print(modlmap[badinds[ind][0],badinds[ind][1]])
        print(modlmap[badinds[ind][0]+10,badinds[ind][1]+10])
        print(p2d[badinds[ind][0],badinds[ind][1]])
        print(p2d[badinds[ind][0]+10,badinds[ind][1]+10])

    p2d[np.logical_and(np.log10(p2d)>llim,modlmap>1000)] = np.nan
    #p2d[sel] = np.nan
    io.power_crop(p2d,150,f"mdet2d{seed}comp.png",lim=[-10,1])
pl.done("detcomp.png")


