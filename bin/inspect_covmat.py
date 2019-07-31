from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
from enlib import bench
import numpy as np
import os,sys
from tilec import pipeline,utils as tutils
from soapack import interfaces as sints

"""
This script will work with a saved covariance matrix to obtain component separated
maps.

The datamodel is only used for beams here.
"""

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("cov_version", type=str,help='Region name.')
parser.add_argument("region", type=str,help='Region name.')
parser.add_argument("arrays", type=str,help='Comma separated list of array names. Array names map to a data specification in data.yml')
args = parser.parse_args()

save_path = sints.dconfig['tilec']['save_path']
covdir = save_path + args.cov_version + "/" + args.region +"/"
mask = enmap.read_map(covdir+"tilec_mask.fits")
modlmap = mask.modlmap()

names = args.arrays.split(',')
narrays = len(names)
arrays = names

aspecs = tutils.ASpecs().get_specs


lmins = []
lmaxs = []
for i,qid in enumerate(arrays):
    dm = sints.models[sints.arrays(qid,'data_model')](region=mask,calibrated=True)
    lmin,lmax,hybrid,radial,friend,cfreq,fgroup = aspecs(qid)
    lmins.append(lmin)
    lmaxs.append(lmax)


i = 0
for aindex1 in range(narrays):
    for aindex2 in range(aindex1,narrays):
        icov = enmap.read_map(covdir+"tilec_hybrid_covariance_%s_%s.hdf" % (names[aindex1],names[aindex2]))
        if aindex1==aindex2: 
            print(names[aindex1],names[aindex2],lmins[aindex1],lmaxs[aindex1])
        #     icov[modlmap<lmins[aindex1]] = 1e200 #np.inf
        #     icov[modlmap>lmaxs[aindex1]] = 1e200 #np.inf
        #     io.plot_img(np.log10(np.fft.fftshift(icov)),"/scratch/r/rbond/msyriac/data/depot/tilec/plots/ap2d_%s_%s.png" % (names[aindex1],names[aindex2]),aspect='auto',lim=[-5,1])
        if i==0: covmat = np.zeros((icov.shape[-2],icov.shape[-1],narrays,narrays))
        i += 1
        covmat[...,aindex1,aindex2] = icov.copy()
        if aindex1 != aindex2: covmat[...,aindex2,aindex1] = icov.copy()

maxval = covmat.max()*100
print(maxval)

for aindex in range(narrays):
    covmat[modlmap<=lmins[aindex],aindex,aindex] = maxval
    covmat[modlmap>=lmaxs[aindex],aindex,aindex] = maxval
    

# sys.exit()
print(covmat.shape)
sel = np.logical_and(modlmap>0,modlmap<250000)
w = np.linalg.eigvalsh(covmat[sel,...])
print(w.shape)
print(w)
print(np.any(w<=0))
for i in range(w.shape[-1]):
    print(modlmap[sel][w[:,i]<=0].reshape(-1))
    print(w[w[:,i]<=0][:,i])




