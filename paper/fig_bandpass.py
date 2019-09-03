from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap
import numpy as np
import os,sys
from soapack import interfaces as sints
from tilec import utils as tutils,fg

fnames = []
qids = "boss_01,boss_02,boss_03,boss_04,p01,p02,p03,p04,p05,p06,p07,p08".split(',')

#pl = io.Plotter(xyscale='loglin',xlabel='$\\nu$',ylabel='$B(\\nu)$')


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
    elif tutils.is_hfi(qid):
        col = 'C1'
    else:
        col = 'C2'

#     pl.add(nu,bp/bp.max(),color=col,lw=2)
# pl._ax.set_xlim(20,800)
# pl.done("fig_bandpass.png")

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
