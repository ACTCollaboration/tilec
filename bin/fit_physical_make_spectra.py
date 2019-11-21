from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap
import numpy as np
import os,sys
import pickle
from szar import foregrounds as fgs
import soapack.interfaces as sints
from tilec import kspace,utils as tutils
from actsims import noise as simnoise
from enlib import bench

"""

We will create a covariance matrix (narrays,narrays,ellmax)
that describes what the power spectra of "residuals" are.
Residual is defined to be what is left over after subtracting
a fiducial lensed CMB and fiducial tSZ spectrum. This procedure
is aimed at producing Gaussian simulations whose total power
matches that of the data without having to model, for example,
the residual foreground power after a complicated source
subtraction procedure, and without having to, for example, add
back sources to make the residual easy to model (thus increasing
noise and bias in a lensing estimator).

We do this by taking the following spectra in the specified
ell ranges,

LFI-30/44 x LFI  20 < ell < 300
LFI-70    x LFI  20 < ell < 900
LFI-30/44 x HFI  20 < ell < 300
LFI-70    x HFI  20 < ell < 900
LFI       x HFI  20 < ell < 300
LFI-30/44 x ACT  -- no residual --
LFI-70    x ACT  1000 < ell < 2000
HFI       x ACT  1000 < ell < 5800
HFI       x HFI  20 < ell < 5800
ACT       x ACT  1000 < ell < 5800




"""


import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("version", type=str,help='Version name.')
parser.add_argument("region", type=str,help='Region name.')
parser.add_argument("arrays", type=str,help='Comma separated list of array names. Array names map to a data specification in data.yml')
parser.add_argument("--mask-version", type=str,  default="padded_v1",help='Mask version')
parser.add_argument("--unsanitized-beam", action='store_true',help='Do not sanitize beam.')
parser.add_argument("--skip-inpainting", action='store_true',help='Do not inpaint.')
parser.add_argument("--uncalibrated", action='store_true',help='Do not use calibration factors.')

args = parser.parse_args()

spath = sints.dconfig['actsims']['fg_res_path'] + "/"+ args.version + "_" + args.region +  "/"

try: os.makedirs(spath)
except: pass


aqids = args.arrays.split(',')
narrays = len(aqids)
qpairs = []
for qid1 in range(narrays):
    for qid2 in range(qid1,narrays):
        qpairs.append((aqids[qid1],aqids[qid2]))


"""
We will MPI parallelize over pairs of arrays. This wastes FFTs, but is much memory efficient. (Each job only
holds 2 arrays in memory).
"""
njobs = len(qpairs)
comm,rank,my_tasks = mpi.distribute(njobs)


mask = sints.get_act_mr3_crosslinked_mask(args.region,
                                          version=args.mask_version,
                                          kind='binary_apod')
shape,wcs = mask.shape,mask.wcs
modlmap = mask.modlmap()
aspecs = tutils.ASpecs().get_specs
region = args.region

fbeam = lambda qname,x: tutils.get_kbeam(qname,x,sanitize=not(args.unsanitized_beam),planck_pixwin=True)
nbin_edges = np.arange(20,8000,100)
nbinner = stats.bin2D(modlmap,nbin_edges)
ncents = nbinner.centers

cbin_edges = np.arange(20,8000,40)
cbinner = stats.bin2D(modlmap,cbin_edges)

for task in my_tasks:
    qids = qpairs[task]
    print("Rank %d doing task %d for array %s x %s ..." % (rank,task,qids[0],qids[1]))
    qid1,qid2 = qids
    do_radial_fit = []
    friends = []
    freqs = []
    rfit_wnoise_widths = []
    kdiffs = []
    kcoadds = []
    wins = []
    for i,qid in enumerate(qids):
        dm = sints.models[sints.arrays(qid,'data_model')](region=mask,calibrated=not(args.uncalibrated))
        lmin,lmax,hybrid,radial,friend,cfreq,fgroup,wrfit = aspecs(qid)
        assert isinstance(radial,bool)
        do_radial_fit.append(radial)
        if friend is not None: 
            assert len(friend)==1
            friend = friend[0]
        friends.append( friend )
        freqs.append(cfreq)
        rfit_wnoise_widths.append(wrfit)
        if qid==qids[0] and i==1:
            kdiff = kdiffs[0].copy()
            kcoadd = kcoadds[0].copy()
            win = wins[0].copy()
        else:
            kdiff,kcoadd,win = kspace.process(dm,region,qid,mask,
                                              skip_splits=False,
                                              splits_fname=None,
                                              inpaint=not(args.skip_inpainting),fn_beam = lambda x: fbeam(qid,x),verbose=False,plot_inpaint_path=None)

        kdiffs.append(kdiff.copy())
        kcoadds.append(kcoadd.copy())
        wins.append(win.copy())

    if friends[0]==qids[1] or qid1==qid2:
        if qid1!=qid2: assert friends[1]==qids[0]
        correlated = True
        print("Array %s and %s are correlated." % (qids[0],qids[1]))
    else:
        correlated = False
        print("Array %s and %s are not correlated." % (qids[0],qids[1]))


    ccov = np.real(kcoadds[0]*kcoadds[1].conj())/np.mean(mask**2)
    if not(correlated):
        scov = ccov
        n1d = ncents * 0
    else:
        ncov = simnoise.noise_power(kdiffs[0],mask,
                                    kmaps2=kdiffs[1],weights2=mask,
                                    coadd_estimator=True)
        scov = ccov - ncov
        ncents,n1d = nbinner.bin(ncov)


    fbeam1 = lambda x: tutils.get_kbeam(qid1,x,sanitize=True,planck_pixwin=True)
    fbeam2 = lambda x: tutils.get_kbeam(qid2,x,sanitize=True,planck_pixwin=True)

    ccents,s1d = cbinner.bin(scov/fbeam1(modlmap)/fbeam2(modlmap))
    io.save_cols("%sn1d_%s_%s.txt" % (spath,qid1,qid2), (ncents,n1d))
    io.save_cols("%ss1d_%s_%s.txt" % (spath,qid1,qid2), (ccents,s1d))
    


