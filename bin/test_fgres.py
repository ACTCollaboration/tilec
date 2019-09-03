from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from orphics import maps,io,cosmology,mpi,stats
from pixell import enmap,curvedsky
import numpy as np
import os,sys,shutil
from datetime import datetime
from actsims import noise as actnoise
from actsims.util import seed_tracker
from soapack import interfaces as sints
from enlib import bench
import healpy as hp
from tilec import pipeline,utils as tutils

region = 'boss'
#region = 'deep56'
fg_res_version = 'test'
#qids = 'd56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08'.split(',')
#qids = 'boss_01'.split(',')
qids = 'boss_01,boss_02,boss_03,boss_04,p01,p02,p03,p04,p05,p06,p07,p08'.split(',')
#qids = 'boss_04,p01,p02,p03,p04,p05,p06,p07,p08'.split(',')
#qids = 'boss_04,p08'.split(',')
#qids = 'boss_03,boss_04'.split(',')
#qids = 'd56_01,d56_02,d56_03,d56_04,d56_05,d56_06'.split(',')
#qids = 'd56_02,d56_03,d56_04'.split(',')
#qids = 'd56_04,p01,p02,p03,p04,p05,p06,p07,p08'.split(',')

mask = sints.get_act_mr3_crosslinked_mask(region,
                                          kind='binary_apod')

modlmap = mask.modlmap()

jsim = pipeline.JointSim(qids,fg_res_version+"_"+region,bandpassed=True)


jsim0 = pipeline.JointSim(qids,None,bandpassed=True)

comm,rank,my_tasks = mpi.distribute(len(qids))

bin_edges = np.arange(20,8000,20)
binner = stats.bin2D(modlmap,bin_edges)

jsim.update_signal_index(0,set_idx=0)
jsim0.update_signal_index(0,set_idx=0)

for task in my_tasks:

    qid = qids[task]

    with bench.show("signal"):
        signal = jsim.compute_map(mask.shape,mask.wcs,qid,
                                  include_cmb=True,include_tsz=True,
                                  include_fgres=True,sht_beam=True) # !!!

    signal0 = jsim0.compute_map(mask.shape,mask.wcs,qid,
                              include_cmb=True,include_tsz=True,
                              include_fgres=True,sht_beam=True)

    enmap.write_map(os.environ['WORK']+"/temp_sig_%s.fits" % qid,signal[0])
    enmap.write_map(os.environ['WORK']+"/temp_sig0_%s.fits" % qid,signal0[0])

comm.Barrier()
if rank==0:
    print("Processing...")
    kmaps = []
    kmaps0 = []
    for qid in qids:
        sig = enmap.read_map(os.environ['WORK']+"/temp_sig_%s.fits" % qid)
        sig0 = enmap.read_map(os.environ['WORK']+"/temp_sig0_%s.fits" % qid)
        print(sig.shape)

        kmaps.append(enmap.fft(sig*mask,normalize='phys'))
        print(kmaps[-1].shape)
        kmaps0.append(enmap.fft(sig0*mask,normalize='phys'))



    narrays = len(qids)
    for i in range(narrays):
        for j in range(i,narrays):
            qid1 = qids[i]
            qid2 = qids[j]

            cents,c = binner.bin(np.real(kmaps[i] * kmaps[j].conj()))
            cents,c0 = binner.bin(np.real(kmaps0[i] * kmaps0[j].conj()))

            pl = io.Plotter(xyscale='linlog',scalefn=lambda x: x**2./2./np.pi,xlabel='l',ylabel='D')
            pl.add(cents,c)
            pl.add(cents,c0,ls='--')
            pl.done(os.environ['WORK'] + "/fgcomp_%s_%s_%s.png" % (region,qid1,qid2))



# print(jsim.cfgres.shape)
# #alm = curvedsky.rand_alm_healpy(jsim.cfgres, lmax=6000)
# alm = curvedsky.rand_alm(jsim.cfgres, lmax=6000)
# imap = hp.alm2map(alm, nside=2048) 
# ocls = hp.anafast(imap,iter=0)
# print(ocls.shape)
# ls = np.arange(len(ocls[0]))
# pl = io.Plotter(xyscale='linlog',scalefn=lambda x: x**2./2./np.pi,xlabel='l',ylabel='D')
# for i in range(ocls.shape[0]):
#     pl.add(ls,ocls[i],marker="o",ls="none")
# pl.done(os.environ['WORK'] + "/fg_hp_all.png")

# pl = io.Plotter(xyscale='linlog',scalefn=lambda x: x**2./2./np.pi,xlabel='l',ylabel='D')
# for i in range(3):
#     for j in range(i,3):
#         ls = np.arange(len(jsim.cfgres[i,j]))
#         pl.add(ls,jsim.cfgres[i,j])
# pl.done(os.environ['WORK'] + "/fg_hp_all2.png")




# sys.exit()
