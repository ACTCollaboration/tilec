from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
from orphics import maps,io,cosmology
from pixell import enmap
import numpy as np
import os,sys
from soapack import interfaces as sints
from tilec import utils as tutils,fg

fnames = []
qids = "boss_01,boss_02,boss_04".split(',')

ftsize = 18
pl = io.Plotter(xyscale='loglin',xlabel='$\\nu$ (GHz)',ylabel='$T(\\nu)$',figsize=(10,3),ftsize=ftsize)

lfi_done = False
hfi_done = False
act_done = False
for qid in qids:
    dm = sints.models[sints.arrays(qid,'data_model')]()

    if dm.name=='act_mr3':
        season,array1,array2 = sints.arrays(qid,'season'),sints.arrays(qid,'array'),sints.arrays(qid,'freq')
        array = '_'.join([array1,array2])
    elif dm.name=='planck_hybrid':
        season,patch,array = None,None,sints.arrays(qid,'freq')

    fname = "data/"+dm.get_bandpass_file_name(array)
    if fname in fnames: continue
    fnames.append(fname)
    print(fname)
    nu,bp = np.loadtxt(fname,unpack=True,usecols=[0,1])
    
    if tutils.is_lfi(qid):
        col = 'black'
        if not(lfi_done): label = 'LFI'
        else: label = None
        lfi_done = True
    elif tutils.is_hfi(qid):
        col = 'red'
        if not(hfi_done): label = 'HFI'
        else: label = None
        hfi_done = True
    else:
        col = 'blue'
        if not(act_done): label = 'ACT'
        else: label = None
        act_done = True


    bnp = bp/bp.max()
    bnp[bnp<5e-4] = np.nan
    pl.add(nu,bnp,color=None,lw=1,label=sints.arrays(qid,'array'),alpha=1)
pl._ax.set_xlim(110,190)
pl.hline()
#pl._ax.set_xticks([30,44,70,100,150, 217, 353,545])
#pl._ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#pl._ax.get_xaxis().get_major_formatter().labelOnlyBase = False
pl.legend(loc='lower right',labsize=ftsize)
pl.done("fig_act_bandpass.png")

