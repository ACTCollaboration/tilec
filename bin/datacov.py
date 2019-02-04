from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from tilec import utils as tutils,covtools,ilc

c = tutils.Config()

ksplits = []
kcoadds = []
atmosphere = []
lmins = []
lmaxs = []
rfit_lmaxes = []
for ai in range(c.narrays):
    ksplit,kcoadd = c.load(ai)
    ksplits.append(ksplit.copy())
    kcoadds.append(kcoadd.copy())
    nsplits = c.darrays[c.arrays[ai]]['nsplits']
    atmosphere.append( nsplits>2 )
    if nsplits>2:
        rfit_lmaxes.append(8000)
    else:
        rfit_lmaxes.append(6000)
    lmins.append ( c.darrays[c.arrays[ai]]['lmin'] )
    lmaxs.append ( c.darrays[c.arrays[ai]]['lmax'] )
    # io.hplot(enmap.downgrade(enmap.enmap(c.fc.ifft(b).real,c.wcs)*np.sqrt(np.mean(c.mask**2.)),4),"coadd_%d" % ai,range=1200)

wcs = c.wcs
del ksplit,kcoadd,c


Cov = ilc.build_empirical_cov(ksplits,kcoadds,atmosphere,lmins,lmaxs,
                              signal_bin_width=80,
                              signal_interp_order=0,
                              # noise_isotropic=False,
                              dfact=(16,16),
                              rfit_lmaxes=rfit_lmaxes,
                              rfit_wnoise_width=250,
                              rfit_lmin=300,
                              rfit_bin_width=80,
                              auto_for_cross_covariance=True,
                              min_splits=4,
                              fc=None,return_full=False,debug_plots=True,alt=True)

enmap.write_map("datacov.hdf",Cov.data)
