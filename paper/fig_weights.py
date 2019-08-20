from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from soapack import interfaces as sints
from tilec import utils as tutils

region = "deep56"
qids = "d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08".split(',')
#qids = "d56_01,d56_02".split(',')
#qids = "d56_05,p01,p05".split(',')
version = "wtest_map_v1.0.0_rc_joint"

fname = lambda qid: "/scratch/r/rbond/msyriac/data/depot/tilec/%s_%s/tilec_single_tile_%s_cmb_%s_%s_weight.fits" % (version,region,region,version,qid)

bw = 20
bin_edges = np.arange(20,10000,bw)
aspecs = tutils.ASpecs().get_specs


w1ds = []
for i,qid in enumerate(qids):
    weight = enmap.read_map(fname(qid))
    modlmap = weight.modlmap()

    lmin,lmax,hybrid,radial,friend,cfreq,fgroup,wrfit = aspecs(qid)    
    weight[modlmap<lmin] = np.nan
    weight[modlmap>lmax] = np.nan

    if tutils.is_lfi(qid):
        N = 40
    elif tutils.is_hfi(qid):
        N = 350
    else:
        N = 500
    
    if i==0:
        binner = stats.bin2D(modlmap,bin_edges)
    Ny,Nx = weight.shape[-2:]
    M = maps.crop_center(np.fft.fftshift(modlmap),N,int(N*Nx/Ny))
    print(M.max())



    io.plot_img(maps.crop_center(np.fft.fftshift(weight),N,int(N*Nx/Ny)),"%s/weight2d_%s.png" % (os.environ['WORK'],qid),aspect='auto',xlabel='$\\ell_x$',ylabel='$\\ell_y$',arc_width=2*M[0,0])


    cents,w1d = binner.bin(weight)
    w1ds.append(w1d)


pl = io.Plotter(xyscale='loglin',xlabel='$\\ell$',ylabel='$W$')
for i in range(len(qids)):

    qid = qids[i]
    lmin,lmax,hybrid,radial,friend,cfreq,fgroup,wrfit = aspecs(qid)    
    w1d = w1ds[i]
    w1d[cents<lmin] = np.nan
    w1d[cents>lmax] = np.nan


    if tutils.is_lfi(qid):
        ls = ":"
        lab = "LFI %d GHz" % cfreq 
    elif tutils.is_hfi(qid):
        ls = "-"
        lab = "HFI %d GHz" % cfreq 
    else:
        ls = "--"
        aind = qid.split("_")[1]
        lab = "ACT_%s %d GHz" % (aind,cfreq )

    pl.add(cents,w1d,label=lab,ls=ls)
pl._ax.set_xlim(20+bw/2.,10000)
pl.legend(loc='upper right', bbox_to_anchor=(1.45, 1))
pl.done("%s/weight1d.png" % (os.environ['WORK']))
