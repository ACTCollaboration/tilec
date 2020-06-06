from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
from enlib import bench
import numpy as np
import os,sys
from tilec import pipeline, utils as tutils
from soapack import interfaces as sints

"""
This script will work with a saved covariance matrix to obtain component separated
maps.

The datamodel is only used for beams here.
"""

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Following the construction of covariance matrices, this script can be used to obtain ILC solutions.')
parser.add_argument("version", type=str,help='Version name to use for the saved objects.')
parser.add_argument("cov_version", type=str,help='Version name that was used for the saved covariance matrix objects.')
parser.add_argument("region", type=str,help='Region name. e.g. deep56, boss, deep1. Possible names for the soapack ACTmr3 datamodel (used in Madhavacheril et. al. 2019) can be found in the patches attribute of that class.')
parser.add_argument("arrays", type=str,help='Comma separated list of array names (qids). Possible array qids can be found in input/array_specs.csv. The soapack data model mapping corresponding to these qids must exist in soapack/soapack/data/all_arrays.csv.')
parser.add_argument("solutions", type=str,help='Comma separated list of solutions. Each solution is of the form x-y-... where x is solved for and the optionally provided y-,... are deprojected. The x can belong to any of CMB,tSZ and y,z,... can belong to any of CMB,tSZ,CIB.')
parser.add_argument("beams", type=str,help='Comma separated list of beams to convolve the final maps with. Each beam is either a float for FWHM in arcminutes or the name of an array whose beam will be used. The number of elements in the list must be the same as for the solutions argument.')
parser.add_argument("-o", "--overwrite", action='store_true',help='Ignore existing version directory and overwrite files.')
parser.add_argument("-e", "--effective-freq", action='store_true',help='Ignore bandpass files and use effective frequencies.')
parser.add_argument("--chunk-size",     type=int,  default=5000000,help="Number of elements in chunk for ILC calculation. Larger may be faster but takes more memory.")
parser.add_argument("--maxval",     type=float,  default=700000,help="The (large) value to set for regions of the covmat where the noise should nominally be infinite.")
parser.add_argument("--ccor-exp",     type=float,  default=-1,help="The exponent of the scaling relation between beam FWHM and frequency used in the color corrections. The default of -1 is for diffraction limited optics.")
parser.add_argument("--pa1-shift",     type=float,  default=None,help="Shift the ACT PA1 bandpasses by the specified GHz.")
parser.add_argument("--pa2-shift",     type=float,  default=None,help="Shift the ACT PA2 bandpasses by the specified GHz.")
parser.add_argument("--pa3-150-shift",     type=float,  default=None,help="Shift the ACT PA3 150 GHz bandpasses by the specified GHz.")
parser.add_argument("--pa3-090-shift",     type=float,  default=None,help="Shift the ACT PA3 90 GHz bandpasses by the specified GHz.")
parser.add_argument("--beam-version", type=str,  default=None,help='Beam version name')
parser.add_argument("--unsanitized-beam", action='store_true',help='Do not sanitize beam.')
parser.add_argument("--no-act-color-correction", action='store_true',help='Do not color correct ACT arrays in a scale dependent way.')

args = parser.parse_args()

print("Command line arguments are %s." % args)
tutils.validate_args(args.solutions,args.beams)

pipeline.build_and_save_ilc(args.arrays,args.region,args.version,args.cov_version,args.beam_version,
                            args.solutions,args.beams,args.chunk_size,
                            args.effective_freq,args.overwrite,args.maxval,unsanitized_beam=args.unsanitized_beam,do_weights=True,
                            pa1_shift = args.pa1_shift,
                            pa2_shift = args.pa2_shift,
                            pa3_150_shift = args.pa3_150_shift,
                            pa3_090_shift = args.pa3_090_shift,
                            no_act_color_correction=args.no_act_color_correction,ccor_exp=args.ccor_exp)
