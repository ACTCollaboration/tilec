from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap
import numpy as np
import os,sys
from tilec import utils as tutils,ilc
from orphics.stats import correlated_hybrid_matrix,cov2corr

aspecs = tutils.ASpecs().get_specs

# qids = "d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08".split(',')
# rads = [0.5,0.4,0.3,0.6,0.65,0.6,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9]
# fs = [146,147,148,149,150,93,30,44,70,100,143,217,353,545]
# fpath = "/scratch/r/rbond/msyriac/data/depot/actsims/fg_res/fgfit_deep56"

qids = "boss_01,boss_02,boss_03,boss_04,p01,p02,p03,p04,p05,p06,p07,p08".split(',')
rads = [0.5,0.4,0.3,0.6,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9]
fs = [146,147,150,93,30,44,70,100,143,217,353,545]
fpath = "/scratch/r/rbond/msyriac/data/depot/actsims/fg_res/fgfit_boss"


narrays = len(qids)



#pl = io.Plotter(xyscale='linlog',scalefn=lambda x: x**2./2./np.pi,xlabel='l',ylabel='D')
pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='C')

c = 0



for i in range(narrays):
    for j in range(i,narrays):
        qid1 = qids[i]
        qid2 = qids[j]

        # if "p02" not in [qid1,qid2]: continue # !!!!
        # if i!=j: continue # !!!

        ls,s1d = np.loadtxt(f"{fpath}/s1d_{qid1}_{qid2}.txt",unpack=True)

        ells,f1d = np.loadtxt(f"{fpath}/fgcov_{qid1}_{qid2}.txt",unpack=True)

        if c==0:
            lmax = ells.size
            cfgres = np.zeros((narrays,narrays,lmax))
            tfgres = np.zeros((narrays,narrays,lmax))
            

        lmin,_,hybrid,radial,friend,f1,fgroup,wrfit = aspecs(qid1)
        lmin,_,hybrid,radial,friend,f2,fgroup,wrfit = aspecs(qid2)

        f1 = fs[i]
        f2 = fs[j]

        fbeam1 = lambda x: tutils.get_kbeam(qid1,x,sanitize=True,planck_pixwin=True)
        fbeam2 = lambda x: tutils.get_kbeam(qid2,x,sanitize=True,planck_pixwin=True)
    
        stheory = ilc.CTheory(ls)
        #s1d = s1d/fbeam1(ls)/fbeam2(ls) - stheory.get_theory_cls(f1,f2,a_cmb=1,a_gal=0,a_cibp=0,a_cibc=0,a_radps=0,a_ksz=0,a_tsz=1)
        s1d = s1d - stheory.get_theory_cls(f1,f2,a_cmb=1,a_gal=0,a_cibp=0,a_cibc=0,a_radps=0,a_ksz=0,a_tsz=1)
        pl.add(ls,s1d,marker='o',ls='none',color=f"C{c}",markersize=3)


        pl.add(ells,f1d,color=f"C{c}",lw=1,label=f'{qid1} x {qid2}')
        cfgres[i,j] = f1d.copy()[:lmax]
        cfgres[j,i] = f1d.copy()[:lmax]

        ftheory = ilc.CTheory(ells)
        t1d = ftheory.get_theory_cls(f1,f2,a_cmb=0,a_gal=0.8,a_cibp=1,a_cibc=1,a_radps=rads[i]*rads[j],a_ksz=1,a_tsz=0)
        pl.add(ells,t1d,color=f"C{c}",lw=1,ls='--')
        tfgres[i,j] = t1d.copy()[:lmax]
        tfgres[j,i] = t1d.copy()[:lmax]

        c+=1
pl._ax.set_xlim(2,5800)
#pl._ax.set_xlim(2,300) # !!!
pl.done("debugfits.png")
    


corr = cov2corr(cfgres)
tcorr = cov2corr(tfgres)
corr[~np.isfinite(corr)] = 0
tcorr[~np.isfinite(tcorr)] = 0
assert np.all(tcorr<=1)
hmat = correlated_hybrid_matrix(cfgres,theory_corr=tcorr)
hmat[~np.isfinite(hmat)] = 0
hcorr = cov2corr(hmat)
hcorr[~np.isfinite(hcorr)] = 0
print(hcorr.max())
assert np.all(hcorr<=(1+1e-5))
sample_ells = [500,1000]


for sell in sample_ells:
    
    cmat = cfgres

    print("==== data")
    eigs = np.linalg.eigh(cfgres[:,:,sell])[0]
    print(sell)
    print(corr[:,:,sell])
    print(eigs)


    print("==== theory")
    eigs = np.linalg.eigh(tfgres[:,:,sell])[0]
    print(sell)
    print(tcorr[:,:,sell])
    print(eigs)
    # assert np.all(eigs>0)


    print("==== hybrid")
    eigs = np.linalg.eigh(hmat[:,:,sell])[0]
    print(sell)
    print(hcorr[:,:,sell])
    print(eigs)
    # assert np.all(eigs>0)



pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='C')
for i in range(narrays):
    for j in range(i,narrays):
        qid1 = qids[i]
        qid2 = qids[j]

        # if "p02" not in [qid1,qid2]: continue # !!!!

        f1d = cfgres[i,j]
        t1d = tfgres[i,j]
        h1d = hmat[i,j]
        pl.add(ells[:lmax],f1d,color=f"C{c}",lw=1,label=f'{qid1} x {qid2}')
        pl.add(ells[:lmax],t1d,color=f"C{c}",lw=1,ls='--')
        pl.add(ells[:lmax],h1d,color=f"C{c}",lw=2,ls=':')
        io.save_cols(f"{fpath}/tfgcov_{qid1}_{qid2}.txt",(ells,t1d))
        io.save_cols(f"{fpath}/hfgcov_{qid1}_{qid2}.txt",(ells,h1d))

        c+=1
pl._ax.set_xlim(2,5800)
pl.done("debugfits2.png")
