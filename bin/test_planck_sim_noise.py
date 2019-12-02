from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from orphics import maps,io,cosmology,mpi,stats
from pixell import enmap
import numpy as np
import os,sys,shutil
from datetime import datetime
from actsims import noise as actnoise
from actsims.util import seed_tracker
from soapack import interfaces as sints
from enlib import bench
from tilec import pipeline,utils as tutils,kspace

region = 'deep56'
arrays = ['p04']
sim_version = "v6.2.0_calibrated_mask_version_padded_v1"

sim_index = 0
set_id = 0

mask = sints.get_act_mr3_crosslinked_mask(region)
w2 = np.mean(mask**2.)


ngen = {}
ngen['planck_hybrid'] = actnoise.NoiseGen(sim_version,model="planck_hybrid",extract_region=mask,ncache=0,verbose=True)
dmname = 'planck_hybrid'

jsim = pipeline.JointSim(arrays,None,
                         bandpassed=True,no_act_color_correction=False,
                         ccor_exp=-1)


jsim.update_signal_index(sim_index,set_idx=set_id)

qid = arrays[0]
signal = jsim.compute_map(mask.shape,mask.wcs,qid,
                          include_cmb=True,include_tsz=True,
                          include_fgres=False,sht_beam=True)


season,arrayname = None,sints.arrays(qid,'freq')
patch = region
farray = arrayname.split('_')[0]
noise_seed = seed_tracker.get_noise_seed(set_id, sim_index, ngen[dmname].dm, season, patch, farray, None)
noise,ivars = ngen[dmname].generate_sim(season=season,patch=patch,array=farray,seed=noise_seed,apply_ivar=False)
ivars = ivars[0]
noise = noise[0]

splits = actnoise.apply_ivar_window(signal[None,None]+noise[None],ivars[None])
nsplits = actnoise.apply_ivar_window(noise[None],ivars[None])
enmap.write_map("temp_planck_sim.fits",splits[0])

print(signal.shape,noise.shape,ivars.shape,splits.shape)

fbeam = lambda qname,x: tutils.get_kbeam(qname,x,sanitize=True,planck_pixwin=True)


kdiff,kcoadd,win = kspace.process(ngen[dmname].dm,region,qid,mask,
                                  skip_splits=False,
                                  splits_fname="temp_planck_sim.fits",
                                  inpaint=False,fn_beam = lambda x: fbeam(qid,x),
                                  plot_inpaint_path = None,
                                  split_set=None)

from actsims import noise as simnoise

ncov = simnoise.noise_power(kdiff,mask*ivars[:,0]/ivars[:,0].sum(axis=0),
                            kmaps2=kdiff,weights2=mask*ivars[:,0]/ivars[:,0].sum(axis=0),
                            coadd_estimator=True)


# ncov = simnoise.noise_power(kdiff,mask,
#                             kmaps2=kdiff,weights2=mask,
#                             coadd_estimator=True)

 
modlmap = mask.modlmap()
bin_edges = np.arange(20,6000,20)
binner = stats.bin2D(modlmap,bin_edges)
def pow(x,y=None):
    k = enmap.fft(x,normalize='phys')
    ky = enmap.fft(y,normalize='phys') if y is not None else k
    p = (k*ky.conj()).real/w2
    cents,p1d = binner.bin(p)
    return p,cents,p1d

tsignal = signal[0]
tsplits = splits[0,:,0,...]
tnsplits = nsplits[0,:,0,...]
tivars = ivars[:,0,...]

tcoadd,_ = tutils.coadd(tsplits,tivars)
tncoadd,_ = tutils.coadd(tnsplits,tivars)

tnoise = tcoadd - tsignal

print(tcoadd.shape,tsignal.shape,tnoise.shape)

_,cents,s1d = pow(tsignal*mask)
_,cents,n1d = pow(tnoise*mask)
_,cents,n1d0 = pow(tncoadd*mask)

cents,n1d_est = binner.bin(ncov)

pl = io.Plotter("Dell")
pl.add(cents,s1d,lw=1)
pl.add(cents,n1d,ls="--",lw=1)
# pl.add(cents,n1d0,ls="--",lw=1)
pl.add(cents,n1d_est,ls=":",lw=1)
pl._ax.set_ylim(1e1,1e4)
pl.done("plsim_det.png")


pl = io.Plotter(ylabel="r",xyscale='linlin',xlabel='l')
pl.add(cents,n1d_est/n1d,ls="--",lw=1)
pl.hline(y=1)
pl.done("plsim_det_rat.png")
