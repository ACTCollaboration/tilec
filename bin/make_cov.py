import matplotlib
matplotlib.use('Agg')
from tilec import kspace,ilc,pipeline
import numpy as np
import os,sys
from pixell import enmap
from enlib import bench
from orphics import io,maps
from soapack import interfaces as sints

"""

This script produces an empirical covariance matrix
from Planck and ACT data.

TODO : make all disk operations thread safe and respect sim number

"""

pdefaults = io.config_from_yaml("input/cov_defaults.yml")['cov']

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Calculate and save covariance matrices for the ILC. Most of these arguments dont need to be specified, in which case they are loaded from input/cov_defaults.yml.')
parser.add_argument("version", type=str,help='Version name to use for the saved objects. The same name has to be provided in the cov_version argument of the make_ilc step that follows this.')
parser.add_argument("region", type=str,help='Region name. e.g. deep56, boss, deep1. Possible names for the soapack ACTmr3 datamodel (used in Madhavacheril et. al. 2019) can be found in the patches attribute of that class.')
parser.add_argument("arrays", type=str,help='Comma separated list of array names (qids). Possible array qids can be found in input/array_specs.csv. The soapack data model mapping corresponding to these qids must exist in soapack/soapack/data/all_arrays.csv.')
parser.add_argument("--mask-version", type=str,  default="padded_v1",help='Mask version')
parser.add_argument("-o", "--overwrite", action='store_true',help='Ignore existing version directory and overwrite files.')
parser.add_argument("-m", "--memory-intensive", action='store_true',help='Do not save FFTs to scratch disk. Can be faster, but very memory intensive.')
parser.add_argument("--unsanitized-beam", action='store_true',help='Do not sanitize beam.')
parser.add_argument("--skip-inpainting", action='store_true',help='Do not inpaint maps.')
parser.add_argument("--uncalibrated", action='store_true',help='Do not use calibration factors.')
parser.add_argument("--isotropic-override", action='store_true',help='Force all noise power spectra to be isotropic.')
parser.add_argument("--theory",     type=str,  default="none",help="Use a theory covariance matrix instead.")
parser.add_argument("--signal-bin-width",     type=int,  default=pdefaults['signal_bin_width'],help="The width of annuli in number of pixels to be used for binning the signal covariance.")
parser.add_argument("--split-set",     type=int,  default=None,help="If you would like to make splits, use 0 to construct the noise model from the 0,1 ACT splits and use 1 to construct from the 2,3 ACT splits.")
parser.add_argument("--signal-interp-order",     type=int,  default=pdefaults['signal_interp_order'],help="The order of interpolation to use when smoothing the signal spectrum.")
parser.add_argument("--delta-ell",     type=int,  default=pdefaults['delta_ell'],help="The block width in Fourier space pixels to use for smoothing the noise power spectrum.")
parser.add_argument("--rfit-lmin",     type=int,  default=pdefaults['rfit_lmin'],help="The minimum ell to include when fitting the radial behaviour of the noise power.")

args = parser.parse_args()
print("Command line arguments are %s." % args)

pipeline.build_and_save_cov(args.arrays,args.region,args.version,args.mask_version,
                            args.signal_bin_width,args.signal_interp_order,args.delta_ell,
                            args.rfit_lmin,
                            args.overwrite,args.memory_intensive,args.uncalibrated,
                            sim_splits=None,skip_inpainting=args.skip_inpainting,
                            theory_signal=args.theory,unsanitized_beam=args.unsanitized_beam,plot_inpaint=True,
                            isotropic_override=args.isotropic_override,split_set=args.split_set)
