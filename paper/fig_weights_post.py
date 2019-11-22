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


cols = ["C%d" % i for i in range(30)]
cols.remove('C8')
for comp in ['cmb','comptony']:

    region = "deep56"
    qids = "d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08".split(',')
    version = "map_v1.0.0_rc_joint"
    cversion = "v1.0.0_rc"

    bw = 20
    bin_edges = np.arange(20,10000,bw)
    aspecs = tutils.ASpecs().get_specs


    w1ds = []

    # actmap = {"d56_01":"D1_149","d56_02":"D2_149","d56_03":"D3_149","d56_04":"D4_149","d56_05":"D5_097","d56_06":"D6_149"}
    actmap = {"d56_01":"D56_1_149","d56_02":"D56_2_149","d56_03":"D56_3_149","d56_04":"D56_4_149","d56_05":"D56_5_097","d56_06":"D56_6_149"}

    if comp=='comptony':
        wstr = '$W (1/\mu K \\times 10^7)$'
    else:
        wstr = '$W$ (dimensionless)'

    pl = io.Plotter(xyscale='loglin',xlabel='$\\ell$',ylabel=wstr,ftsize=16)
    for i in range(len(qids)):
        col = cols[i]
        qid = qids[i]
        lmin,lmax,hybrid,radial,friend,cfreq,fgroup,wrfit = aspecs(qid)    


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
        cents,w1d = np.loadtxt("weights_%s_%s_%s.txt" % (comp,version,lab),unpack=True)

        pl.add(cents,w1d*mul,label=lab if comp=='comptony' else None,ls=ls,color=col)
    pl._ax.set_xlim(20+bw/2.,10000)



    pl._ax.yaxis.set_minor_locator(AutoMinorLocator())
    #pl._ax.xaxis.set_minor_locator(AutoMinorLocator())
    pl._ax.tick_params(axis='x',which='both', width=1)
    pl._ax.tick_params(axis='y',which='both', width=1)
    pl._ax.xaxis.grid(True, which='both',alpha=0.3)
    pl._ax.yaxis.grid(True, which='both',alpha=0.3)

    
    font = {'family': 'serif',
            'color':  'black',
            'weight': 'bold',
            'size': 14,
        }


    if comp=='comptony': 
        pl.legend(loc='upper right', bbox_to_anchor=(1.45, 1),labsize=12)
        pl._ax.text(600, 3, "Compton-$y$ weights",fontdict = font)
    elif comp=='cmb': 
        pl._ax.text(600, 4, "CMB+kSZ weights",fontdict = font)

    pl.done(("%s/fig_weight1d_%s_%s" % (os.environ['WORK'],comp,version)).replace('.','_')+".pdf")
