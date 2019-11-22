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
from tilec import fg as tfg,utils as tutils
from soapack import interfaces as sints

nu = np.geomspace(20,1000,1000)
nu0 = 150
alpha = 0.5

#comps = ['CMB','rSZ','tSZ','CIB','mu']
comps = ['CMB','tSZ','CIB']
#comps = ['rSZ','tSZ','mu']
cols = ['C0','C1','C2','C3','C4']
pl = io.Plotter(xyscale='loglog',xlabel='$\\nu$ (GHz)',ylabel='$f(\\nu)$',figsize=(6,4))
aspecs = tutils.ASpecs().get_specs

def add_pl(i,fs,f0,alpha):
    fs = fs/f0
    if np.any(fs<=0):
        nun = nu[fs<=0]
        fn = -fs[fs<=0]
        pl.add(nun,fn,ls='--',color=col,alpha=alpha,zorder=0)

    fout = fs
    fout[fs<=0] = np.nan
    pl.add(nu,fout,color=col,label=comp + " or kSZ" if comp=='CMB' else comp,alpha=alpha,zorder=0)

pl._ax.xaxis.grid(True, which='both',alpha=0.3)
pl._ax.yaxis.grid(True, which='both',alpha=0.5)
pl._ax.set_axisbelow(True)

for i,comp in enumerate(comps):
    fs = tfg.get_mix(nu, comp=comp)
    f0 = np.abs(tfg.get_mix(nu0, comp=comp))
    col = "C%d" % i
    add_pl(i,fs,f0,alpha=1)

    if comp in ['tSZ','CIB']:
        qids = "boss_01,boss_02,boss_03,boss_04,p01,p02,p03,p04,p05,p06,p07,p08".split(',')
        for qid in qids:
            lmin,lmax,hybrid,radial,friend,cfreq,fgroup,wrfit = aspecs(qid)
            dm = sints.models[sints.arrays(qid,'data_model')]()

            if dm.name=='act_mr3':
                season,array1,array2 = sints.arrays(qid,'season'),sints.arrays(qid,'array'),sints.arrays(qid,'freq')
                array = '_'.join([array1,array2])
                marker = 's'
            elif dm.name=='planck_hybrid':
                season,patch,array = None,None,sints.arrays(qid,'freq')
                marker = 'o'

            bps = ["data/"+dm.get_bandpass_file_name(array)]
            fr = tfg.get_mix_bandpassed(bps, comp,normalize_cib=False)[0]
            fc = tfg.get_mix(cfreq, comp=comp)[0]
            if comp=='CIB': print(comp,fr,fc,f0)
            f0 = tfg.get_mix_bandpassed(['data/PA2_avg_passband_wErr_trunc.txt'], comp,normalize_cib=False)[0]
            fs = np.abs(fr/f0)
            if cfreq==93: cfreq = 97
            if cfreq==148: cfreq = 149
            print(cfreq)
            pl._ax.scatter(cfreq,fs,color=col,edgecolors='k',zorder=1,marker=marker)


# pl._ax.set_xticks([30,44,70,100,150, 217, 353,545])
# pl._ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

pl.hline(y=1)
pl._ax.set_ylim(5e-2,400)
pl.done("fig_sed.pdf")
