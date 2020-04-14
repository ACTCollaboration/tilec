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

for comp in ['cmb','comptony']:

    region = "deep56"
    qids = "d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08".split(',')
    #qids = "d56_01,d56_02,d56_03,d56_04,d56_05,d56_06".split(',')
    #qids = "d56_01,d56_02".split(',')
    #qids = "d56_05,p01,p05".split(',')

    # version = "map_v1.0.0_rc_joint"
    # cversion = "v1.0.0_rc"

    # version = "map_v1.1.1_joint"
    # cversion = "v1.1.1"

    version = "map_v1.2.0_joint"
    cversion = "v1.2.0"


    fname = lambda qid: "/scratch/r/rbond/msyriac/data/depot/tilec/v1.2.0_20200324/%s_%s/tilec_single_tile_%s_%s_%s_%s_weight.fits" % (version,region,region,comp,version,qid)
    cname = lambda qid: "/scratch/r/rbond/msyriac/data/depot/tilec/v1.2.0_20200324/%s_%s/tilec_hybrid_covariance_%s_%s.npy" % (cversion,region,qid,qid)

    bw = 20
    bin_edges = np.arange(20,10000,bw)
    aspecs = tutils.ASpecs().get_specs


    w1ds = []
    for i,qid in enumerate(qids):
        weight = enmap.read_map(fname(qid))
        cov = enmap.enmap(np.load(cname(qid)),weight.wcs)
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



        # io.plot_img(maps.crop_center(np.fft.fftshift(weight),N,int(N*Nx/Ny)),"%s/weight2d_%s.pdf" % (os.environ['WORK'],qid),aspect='auto',xlabel='$\\ell_x$',ylabel='$\\ell_y$',arc_width=2*M[0,0])

        # io.plot_img(np.log10(maps.crop_center(np.fft.fftshift(cov),N,int(N*Nx/Ny))),"%s/cov2d_%s.pdf" % (os.environ['WORK'],qid),aspect='auto',xlabel='$\\ell_x$',ylabel='$\\ell_y$',arc_width=2*M[0,0])


        cents,w1d = binner.bin(weight)
        w1ds.append(w1d)


    actmap = {"d56_01":"D56_1_149","d56_02":"D56_2_149","d56_03":"D56_3_149","d56_04":"D56_4_149","d56_05":"D56_5_097","d56_06":"D56_6_149"}

    #pl = io.Plotter(xyscale='loglin',xlabel='$\\ell$',ylabel='$W$')
    if comp=='comptony':
        wstr = '$W (1/\mu K \\times 10^7)$'
    else:
        wstr = '$W$ (dimensionless)'

    pl = io.Plotter(xyscale='loglin',xlabel='$\\ell$',ylabel=wstr,ftsize=16)
    for i in range(len(qids)):

        qid = qids[i]
        lmin,lmax,hybrid,radial,friend,cfreq,fgroup,wrfit = aspecs(qid)    
        w1d = w1ds[i]
        w1d[cents<lmin] = np.nan
        w1d[cents>lmax] = np.nan


        if tutils.is_lfi(qid):
            ls = "-."
            lab = "LFI %d GHz" % cfreq 
        elif tutils.is_hfi(qid):
            ls = "--"
            lab = "HFI %d GHz" % cfreq 
        else:
            ls = "-"
            aind = qid.split("_")[1]
            lab = actmap[qid] #"ACT_%s %d GHz" % (aind,cfreq )
        mul = 1e7 if comp=='comptony' else 1
        io.save_cols("weights_%s_%s_%s.txt" % (comp,version,lab) ,(cents,w1d))
        pl.add(cents,w1d*mul,label=lab,ls=ls)
    pl._ax.set_xlim(20+bw/2.,10000)



    pl._ax.yaxis.set_minor_locator(AutoMinorLocator())
    #pl._ax.xaxis.set_minor_locator(AutoMinorLocator())
    pl._ax.tick_params(axis='x',which='both', width=1)
    pl._ax.tick_params(axis='y',which='both', width=1)
    pl._ax.xaxis.grid(True, which='both',alpha=0.3)
    pl._ax.yaxis.grid(True, which='both',alpha=0.3)


    pl.legend(loc='upper right', bbox_to_anchor=(1.4, 1),labsize=12)
    pl.done(("%s/fig_weight1d_%s_%s" % (os.environ['WORK'],comp,version)).replace('.','_')+".pdf")
