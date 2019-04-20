from tilec import kspace,ilc
import numpy as np
import os,sys
from pixell import enmap
from enlib import bench
from orphics import io,maps
from soapack import interfaces as sints

"""

This script produces an empirical covariance matrix
from Planck and ACT data.


"""

pdefaults = io.config_from_yaml("input/cov_defaults.yml")['cov']

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("version", type=str,help='Version name.')
parser.add_argument("region", type=str,help='Region name.')
parser.add_argument("arrays", type=str,help='Comma separated list of array names. Array names map to a data specification in data.yml')
parser.add_argument("--mask-version", type=str,  default="padded_v1",help='Mask version')
parser.add_argument("--nsim",     type=int,  default=None,help="Number of sims. If not specified, runs on data.")
parser.add_argument("-o", "--overwrite", action='store_true',help='Ignore existing version directory.')
parser.add_argument("-m", "--memory-intensive", action='store_true',help='Do not save FFTs to scratch disk. Can be faster, but very memory intensive.')
parser.add_argument("--signal-bin-width",     type=int,  default=pdefaults['signal_bin_width'],help="A description.")
parser.add_argument("--signal-interp-order",     type=int,  default=pdefaults['signal_interp_order'],help="A description.")
parser.add_argument("--dfact",     type=int,  default=pdefaults['dfact'],help="A description.")
parser.add_argument("--rfit-bin-width",     type=int,  default=pdefaults['rfit_bin_width'],help="A description.")
parser.add_argument("--rfit-wnoise-width",     type=int,  default=pdefaults['rfit_wnoise_width'],help="A description.")
parser.add_argument("--rfit-lmin",     type=int,  default=pdefaults['rfit_lmin'],help="A description.")

args = parser.parse_args()

save_scratch = not(args.memory_intensive)
save_path = sints.dconfig['tilec']['save_path']
scratch_path = sints.dconfig['tilec']['scratch_path']
savedir = save_path + args.version + "/" + args.region +"/"
if save_scratch: scratch = scratch_path + args.version + "/" + args.region +"/"
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
gconfig = io.config_from_yaml("input/data.yml")


mask = sints.get_act_mr3_crosslinked_mask(args.region,
                                          version=args.mask_version,
                                          kind='binary_apod')
shape,wcs = mask.shape,mask.wcs
fc = maps.FourierCalc(shape[-2:],wcs)

with bench.show("ffts"):
    kcoadds = []
    ksplits = []
    wins = []
    lmins = []
    lmaxs = []
    hybrids = []
    save_names = [] # to make sure nothing is overwritten
    friends = {} # what arrays are each correlated with?
    names = []
    for array in args.arrays.split(','):
        ainfo = gconfig[array]
        dm = sints.models[ainfo['data_model']](region=mask)
        name = ainfo['id']
        names.append(name)
        try: friends[name] = ainfo['correlated']
        except: friends[name] = None
        hybrids.append(ainfo['hybrid_average'])
        ksplit,kcoadd,win = kspace.process(dm,args.region,name,fc,mask,ncomp=1,skip_splits=False)
        if save_scratch: 
            kcoadd_name = scratch + "kcoadd_%s.hdf" % array
            ksplit_name = scratch + "ksplit_%s.hdf" % array
            win_name = scratch + "win_%s.hdf" % array
            assert win_name not in save_names
            assert kcoadd_name not in save_names
            assert ksplit_name not in save_names
            enmap.write_map(win_name,win)
            enmap.write_map(kcoadd_name,kcoadd)
            enmap.write_map(ksplit_name,ksplit)
            wins.append(win_name)
            kcoadds.append(kcoadd_name)
            ksplits.append(ksplit_name)
            save_names.append(win_name)
            save_names.append(kcoadd_name)
            save_names.append(ksplit_name)
        else:
            wins.append(win.copy())
            kcoadds.append(kcoadd.copy())
            ksplits.append(ksplit.copy())
        lmins.append(ainfo['lmin'])
        lmaxs.append(ainfo['lmax'])

# Decide what pairs to do hybrid smoothing for
narrays = len(args.arrays.split(','))
anisotropic_pairs = []
for i in range(narrays):
    for j in range(i,narrays):
        name1 = names[i]
        name2 = names[j]
        if (i==j) and (hybrids[i] and hybrids[j]):
            anisotropic_pairs.append((i,j))
            continue
        if (friends[name1] is None) or (friends[name2] is None): continue
        if name2 in friends[name1]:
            assert name1 in friends[name2], "Correlated arrays spec is not consistent."
            anisotropic_pairs.append((i,j))
        
print("Anisotropic pairs: ",anisotropic_pairs)

Cov = ilc.build_empirical_cov(ksplits,kcoadds,wins,mask,lmins,lmaxs,
                              anisotropic_pairs,
                              signal_bin_width=args.signal_bin_width,
                              signal_interp_order=args.signal_interp_order,
                              dfact=(args.dfact,args.dfact),
                              rfit_lmaxes=None,
                              rfit_wnoise_width=args.rfit_wnoise_width,
                              rfit_lmin=args.rfit_lmin,
                              rfit_bin_width=None,
                              fc=fc,return_full=False,
                              verbose=True,
                              debug_plots_loc=savedir)

enmap.write_map("%s/datacov_triangle.hdf" % savedir,Cov.data)
