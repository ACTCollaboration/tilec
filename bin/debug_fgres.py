from __future__ import print_function
import matplotlib.pyplot as plt
from pixell import enmap,curvedsky
import numpy as np
import os,sys
import healpy as hp
from orphics import io
from tilec import utils as tutils

region = 'deep56'
fg_res_version = "fgfit_deep56"
pref = "h"

qids = "d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08".split(',')

ellmax=8101
ells = np.arange(ellmax)
lmax = 8101 # Make this lower to run this faster... ell ~ 1000 is needed to see ACT relevant stuff

fpath = "/scratch/r/rbond/msyriac/data/depot/actsims/fg_res//fgfit_deep56/"
narrays = len(qids)


aspecs = tutils.ASpecs().get_specs


cfgres = np.zeros((narrays,narrays,lmax))
corrs = np.zeros((narrays,narrays,lmax))
for i in range(narrays):
    for j in range(i,narrays):
        qid1 = qids[i]
        qid2 = qids[j]
        clfilename = "%s%sfgcov_%s_%s.txt" % (fpath,pref,qid1,qid2)
        clfilename_alt = "%s%sfgcov_%s_%s.txt" % (fpath,pref,qid2,qid1)
        try:
            ls,cls = np.loadtxt(clfilename,unpack=True)
        except:
            ls,cls = np.loadtxt(clfilename_alt,unpack=True)
        assert np.all(np.isclose(ls,ells))
        cfgres[i,j] = cls.copy()[:lmax]
        if i!=j: 
            cfgres[j,i] = cls.copy()[:lmax]
            # cfgres[j,i] = cfgres[i,j] = cls.copy()[:lmax]*0


        
        flmin1,flmax1,hybrid,radial,friend,cfreq,fgroup,wrfit = aspecs(qid1)
        flmin2,flmax2,hybrid,radial,friend,cfreq,fgroup,wrfit = aspecs(qid2)
        cfgres[i,j][ls<flmin1] = 0
        cfgres[i,j][ls<flmin2] = 0
        cfgres[j,i][ls<flmin1] = 0
        cfgres[j,i][ls<flmin2] = 0

        cfgres[i,j][ls>flmax1] = 0
        cfgres[i,j][ls>flmax2] = 0
        cfgres[j,i][ls>flmax1] = 0
        cfgres[j,i][ls>flmax2] = 0


pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='r')
for i in range(narrays):
    for j in range(i,narrays):
        corrs[i,j] = cfgres[i,j]/np.sqrt(cfgres[i,i]*cfgres[j,j])
        corrs[j,i] = corrs[i,j]
        print(corrs[i,j][np.abs(corrs[i,j])>1])
        ls = np.arange(corrs[i,j].size)
        qid1 = qids[i]
        qid2 = qids[j]
        lab = "--" if "d" in qid1 or "d" in qid2 else "-"
        
        if i!=j: 
            pl.add(ls,corrs[i,j],ls=lab)
pl._ax.set_xlim(10,6500)
pl.hline(y=1)
pl.done("/scratch/r/rbond/msyriac/data/depot/tilec/plots/allfits.png")

#sys.exit()


#lsave = 400
#np.savetxt("/scratch/r/rbond/msyriac/data/for_sigurd/bad_matrix.txt",cfgres[:,:,lsave])
# sys.exit()

fgres_seed = 1
print("done with txt")

print(cfgres.dtype)

alms2 = curvedsky.rand_alm(cfgres, lmax=lmax, seed = fgres_seed)
alms1 = curvedsky.rand_alm_healpy(cfgres, seed = fgres_seed)
print("done")

for i in range(narrays):
    for j in range(i,narrays):
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        ax.set_xlabel('l',fontsize=14)
        ax.set_ylabel('D',fontsize=14)
        ax.set_xscale('linear', nonposx='clip') 
        ax.set_yscale('log', nonposy='clip')

        qid1 = qids[i]
        qid2 = qids[j]

        clth = cfgres[i,j]
        ls = np.arange(clth.size)
        ax.plot(ls,clth*ls**2,color='k',zorder=1,linewidth=1)

        cl1 = hp.alm2cl(alms1[i],alms1[j])
        #cl2 = hp.alm2cl(alms2[i],alms2[j])
        # print(cl1,cl2)
        ls = np.arange(cl1.size)
        ax.plot(ls,cl1*ls**2,zorder=0,linewidth=1,label='healpy')
        #ls = np.arange(cl2.size)
        #ax.plot(ls,cl2*ls**2,zorder=0,linewidth=1,label='pixell')

        ax.set_xlim(10,6500)
        ax.set_ylim(1,1e8)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels,frameon = 1)

        fname = "/scratch/r/rbond/msyriac/data/depot/tilec/plots/debugfits_%s_%s_%s.png" % (region,qid1,qid2)
        fig.savefig(fname,bbox_inches='tight')
        print("saved fig %s " % fname)

