from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap,curvedsky
from enlib import bench
import numpy as np
import os,sys,shutil
from tilec import fg as tfg,ilc,kspace,utils as tutils
from soapack import interfaces as sints
from szar import foregrounds as fgs
import healpy as hp
from actsims.util import seed_tracker

from scipy.ndimage.filters import gaussian_filter as smooth

pm = enmap.read_map("/scratch/r/rbond/msyriac/data/planck/data/pr2/COM_Mask_Lensing_2048_R2.00_car_deep56_interp_order0.fits")
wcs = pm.wcs
pmask = enmap.enmap(smooth(pm,sigma=10),wcs)

region = "deep56"
arrays = "p04,p05,p06,p07,p08".split(',')
mask = sints.get_act_mr3_crosslinked_mask(region) * pmask

# io.hplot(mask,"pmask.png")
# sys.exit()

w2 = np.mean(mask**2.)
modlmap = mask.modlmap()

lmax = 5500
ells = np.arange(0,lmax,1)
ctheory = ilc.CTheory(ells)
aspecs = tutils.ASpecs().get_specs
bin_edges = np.arange(100,lmax,100)
binner = stats.bin2D(modlmap,bin_edges)

pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='D',scalefn = lambda x: x**2./2./np.pi)

for i,qid in enumerate(arrays):
    kcoadds = []
    for splitnum in range(2):
        dm = sints.models[sints.arrays(qid,'data_model')](region=mask,calibrated=True)
        lmin,lmax,hybrid,radial,friend,cfreq,fgroup,wrfit = aspecs(qid)
        cltt = ctheory.get_theory_cls(cfreq,cfreq,a_cmb=1,a_gal=0.8)
        _,kcoadd,_ = kspace.process(dm,region,qid,mask,
                                    skip_splits=True,
                                    splits_fname=None,
                                    inpaint=False,fn_beam = None,
                                    plot_inpaint_path = None,
                                    split_set=splitnum)
        kbeam = tutils.get_kbeam(qid,modlmap,sanitize=False,planck_pixwin=True)
        kcoadds.append(kcoadd/kbeam)

    power = (kcoadds[0]*kcoadds[1].conj()).real / w2
    cents,p1d = binner.bin(power)

    pl.add(cents,p1d,color="C%d" % i,label=str(cfreq))
    pl.add(ells,cltt,color="C%d" % i,ls='--')

pl._ax.set_ylim(1,1e8)
pl.done("planckspec.png")
