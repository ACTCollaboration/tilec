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
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("version", type=str,help='Version name.')
parser.add_argument("region", type=str,help='Region name.')
parser.add_argument("arrays", type=str,help='Comma separated list of array names. Array names map to a data specification in data.yml')
parser.add_argument("--mask-version", type=str,  default="padded_v1",help='Mask version')
parser.add_argument("-o", "--overwrite", action='store_true',help='Ignore existing version directory.')
parser.add_argument("-m", "--memory-intensive", action='store_true',help='Do not save FFTs to scratch disk. Can be faster, but very memory intensive.')
parser.add_argument("--skip-inpainting", action='store_true',help='Do not inpaint.')
parser.add_argument("--uncalibrated", action='store_true',help='Do not use calibration factors.')
parser.add_argument("--theory",     type=str,  default="none",help="A description.")
parser.add_argument("--signal-bin-width",     type=int,  default=pdefaults['signal_bin_width'],help="A description.")
parser.add_argument("--signal-interp-order",     type=int,  default=pdefaults['signal_interp_order'],help="A description.")
parser.add_argument("--delta-ell",     type=int,  default=pdefaults['delta_ell'],help="A description.")
parser.add_argument("--rfit-bin-width",     type=int,  default=pdefaults['rfit_bin_width'],help="A description.")
parser.add_argument("--rfit-wnoise-width",     type=int,  default=pdefaults['rfit_wnoise_width'],help="A description.")
parser.add_argument("--rfit-lmin",     type=int,  default=pdefaults['rfit_lmin'],help="A description.")

args = parser.parse_args()

pipeline.build_and_save_cov(args.arrays,args.region,args.version,args.mask_version,
                            args.signal_bin_width,args.signal_interp_order,args.delta_ell,
                            args.rfit_wnoise_width,args.rfit_lmin,
                            args.overwrite,args.memory_intensive,args.uncalibrated,
                            sim_splits=None,skip_inpainting=args.skip_inpainting,theory_signal=args.theory)
