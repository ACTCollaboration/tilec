from tilec import datamodel,ilc
import numpy as np
import os,sys
from pixell import enmap

"""

This script produces an empirical covariance matrix
from Planck and ACT data.


"""

signal_bin_width=80
signal_interp_order=0
dfact=16
rfit_bin_width = 80
rfit_wnoise_width=250
rfit_lmin=300


import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("version", type=str,help='Region name.')
parser.add_argument("region", type=str,help='Region name.')
parser.add_argument("arrays", type=str,help='Comma separated list of array names. Array names map to a data specification in data.yml')
parser.add_argument("--nsim",     type=int,  default=None,help="A description.")
# parser.add_argument("-N", "--nsim",     type=int,  default=None,help="A description.")
# parser.add_argument("-f", "--flag", action='store_true',help='A flag.')
parser.add_argument("-o", "--overwrite", action='store_true',help='Ignore existing version directory.')
parser.add_argument("-m", "--memory-intensive", action='store_true',help='Do not save FFTs to scratch disk. Can be faster, but very memory intensive.')
args = parser.parse_args()

save_scratch = not(args.memory_intensive)
gconfig = datamodel.gconfig
savedir = datamodel.paths['save'] + args.version + "/" + args.region +"/"
if save_scratch: scratch = datamodel.paths['scratch'] + args.version + "/" + args.region +"/"
if not(args.overwrite):
    assert not(os.path.exists(savedir)), \
   "This version already exists on disk. Please use a different version identifier."
try: os.makedirs(savedir)
except:
    if args.overwrite: pass
    else: raise
if save_scratch:     
    try: os.makedirs(scratch)
    except: pass

kcoadds = []
ksplits = []
lmins = []
lmaxs = []
anisotropic_pairs = []
save_names = [] # to make sure nothing is overwritten
for array in args.arrays.split(','):
    array_datamodel = gconfig[array]['data_model']
    dm = datamodel.datamodels[array_datamodel](args.region,gconfig[array])
    ksplit,kcoadd = dm.process()
    if save_scratch: 
        kcoadd_name = scratch + "kcoadd_%s.hdf" % array
        ksplit_name = scratch + "ksplit_%s.hdf" % array
        enmap.write_map(kcoadd_name,kcoadd)
        enmap.write_map(ksplit_name,ksplit)
        kcoadds.append(kcoadd_name)
        ksplits.append(ksplit_name)
    else:
        kcoadds.append(kcoadd.copy())
        ksplits.append(ksplit.copy())
    lmins.append(dm.c['lmin'])
    lmaxs.append(dm.c['lmax'])
    anisotropic_pairs.append(dm.c['hybrid_average'])




Cov = ilc.build_empirical_cov(ksplits,kcoadds,lmins,lmaxs,
                              anisotropic_pairs,
                              signal_bin_width=signal_bin_width,
                              signal_interp_order=signal_interp_order,
                              dfact=(dfact,dfact),
                              rfit_lmaxes=None,
                              rfit_wnoise_width=rfit_wnoise_width,
                              rfit_lmin=rfit_lmin,
                              rfit_bin_width=None,
                              fc=dm.fc,return_full=False,
                              verbose=True,
                              debug_plots_loc=savedir)

enmap.write_map("%s/datacov_triangle.hdf" % savedir,Cov.data)
