from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "stix"
from orphics import maps,io,cosmology
from pixell import enmap
import numpy as np
import os,sys
from soapack import interfaces as sints
from tilec import utils as tutils,fg

fnames = []
qids = "boss_01,boss_02,boss_03,boss_04,p01,p02,p03,p04,p05,p06,p07,p08".split(',')

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
        col = 'C0'
        if not(lfi_done): label = 'LFI'
        else: label = None
        lfi_done = True
    elif tutils.is_hfi(qid):
        col = 'C1'
        if not(hfi_done): label = 'HFI'
        else: label = None
        hfi_done = True
    else:
        col = 'C2'
        if not(act_done): label = 'ACT'
        else: label = None
        act_done = True


    bnp = bp/bp.max()
    bnp[bnp<5e-4] = np.nan
    pl.add(nu,bnp,color=col,lw=2,label=label)
pl._ax.set_xlim(20,800)
pl.hline()
pl._ax.set_xticks([30,44,70,100,150, 217, 353,545])
pl._ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#pl._ax.get_xaxis().get_major_formatter().labelOnlyBase = False
pl.legend(loc='lower right',labsize=ftsize)
pl.done("fig_bandpass.pdf")

sys.exit()

comp = 'tSZ'
mix = fg.get_mix_bandpassed(fnames, comp)
mixp = fg.get_mix_bandpassed(fnames, comp,shifts=[2.4,2.4,1.5,2.4] + [0]*8)
mixn = fg.get_mix_bandpassed(fnames, comp,shifts=[-2.4,-2.4,-1.5,-2.4] + [0]*8)

diff = (mixp - mix)*100./mix
diff2 = (mixn - mix)*100./mix
diff3 = (mixp - mixn)*100./mix
print(diff,diff2,diff3)

sys.exit()
dm = sints.ACTmr3()
for season in dm.cals.keys():
    for patch in dm.cals[season].keys():
        for array in dm.cals[season][patch].keys():
            cal = dm.cals[season][patch][array]['cal']
            cal_err = dm.cals[season][patch][array]['cal_err']
            print(season,patch,array,cal_err*100./cal)
