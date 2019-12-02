from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap,utils
import numpy as np
import os,sys
import soapack.interfaces as sints
from tilec import kspace,utils as tutils
from actsims import noise as simnoise
from enlib import bench

region = 'deep56'
version = "noLFI_nohigh_test"
qids = "d56_04,d56_05,d56_06,p04,p05,p06".split(',')

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

seeds = [12,13]

#qids = "d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08".split(',')
#qids = "d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p04,p05,p06,p07".split(',')
#qids = "d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p03,p04,p05,p06,p07,p08".split(',')
#qids = "p01,p02,p03,p04,p05,p06,p07,p08".split(',')
narrays = len(qids)
#badpix = [7, 6645]
#goodpix = [7+10, 6645+10]

badpix = [3, 195]
goodpix = [3+10, 195+10]

print(modlmap[badpix[0],badpix[1]])
print(lmap[0][badpix[0],badpix[1]])
print(lmap[1][badpix[0],badpix[1]])

pl = io.Plotter(xyscale='linlin',xlabel='a',ylabel='P')
vals = {}
gvals = {}
for seed in seeds:
    vals[seed] = []
    gvals[seed] = []
    gcov = np.zeros((narrays,narrays))
    bcov = np.zeros((narrays,narrays))
    c = 0
    
    for i in range(narrays):
        for j in range(i,narrays):
            qid1 = qids[i]
            qid2 = qids[j]
        
            cov = np.load(f"{droot}test_sim_galtest_nofg_{version}_00_00{seed}_{region}/tilec_hybrid_covariance_{qid1}_{qid2}.npy")
            #cov = np.load(f"{droot}test_sim_galtest_nofg_00_00{seed}_{region}/tilec_hybrid_covariance_{qid1}_{qid2}.npy")
            bcov[i,j] = cov[badpix[0],badpix[1]]
            bcov[j,i] = cov[badpix[0],badpix[1]]
            gcov[i,j] = cov[goodpix[0],goodpix[1]]
            gcov[j,i] = cov[goodpix[0],goodpix[1]]
            c = c+ 1
            pl._ax.scatter(c,bcov[i,j],color=f"C{seed}",marker="o")
            vals[seed].append(bcov[i,j])
            gvals[seed].append(gcov[i,j])
            if i==j: 
                print(qid1,gcov[i,j])
                if qid1=='p03': gcov[i,j] = bcov[i,j] = 0
                if gcov[i,j]==0.: 
                    gcov[i,j] = 700000
                if bcov[i,j]==0.: bcov[i,j] = 700000


    print(np.diagonal(bcov))
    print(bcov)
    np.savetxt(f"bcov_{seed}.txt",bcov,delimiter=',')
    np.savetxt(f"gcov_{seed}.txt",gcov,delimiter=',')
    gcorr = stats.cov2corr(gcov)
    bcorr = stats.cov2corr(bcov)
    print(gcorr.min(),gcorr.max())
    print(bcorr.min(),bcorr.max())
    io.plot_img(gcorr,f"det_gcov_{seed}.png",flip=False,lim=[0.5,1])
    io.plot_img(bcorr,f"det_bcov_{seed}.png",flip=False,lim=[0.5,1])
    print(seed)
    gi = np.linalg.inv(gcov)
    bi = np.linalg.inv(bcov)
    bi2 = utils.eigpow(bcov,-1)
    print(np.linalg.eigh(gcov)[0])
    print(np.linalg.eigh(bcov)[0])

pl.done("detscatter.png")


pl = io.Plotter(xyscale='linlin',xlabel='a',ylabel='r')
c = 0
    
for i in range(narrays):
    for j in range(i,narrays):
        v1 = vals[12][c]
        v2 = vals[13][c]
        c = c + 1
        qid1 = qids[i]
        qid2 = qids[j]
        if qid1=='d56_05' or qid2=='d56_05':
            marker = 'd'
        else:
            marker = 'o'
        
        pl._ax.scatter(c,(v1-v2)/v2,label=f'{qids[i]} x {qids[j]}',marker=marker)
pl.hline()
pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pl.done("detrelscat.png")


pl = io.Plotter(xyscale='linlin',xlabel='a',ylabel='r')
c = 0
    
for i in range(narrays):
    for j in range(i,narrays):
        v1 = gvals[12][c]
        v2 = gvals[13][c]
        c = c + 1
        qid1 = qids[i]
        qid2 = qids[j]
        if qid1=='d56_05' or qid2=='d56_05':
            marker = 'd'
        else:
            marker = 'o'
        
        pl._ax.scatter(c,(v1-v2)/v2,label=f'{qids[i]} x {qids[j]}',marker=marker)
pl.hline()
pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pl.done("detgrelscat.png")
