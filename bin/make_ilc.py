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
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("version", type=str,help='Region name.')
parser.add_argument("cov_version", type=str,help='Region name.')
parser.add_argument("region", type=str,help='Region name.')
parser.add_argument("arrays", type=str,help='Comma separated list of array names. Array names map to a data specification in data.yml')
parser.add_argument("solutions", type=str,help='Comma separated list of solutions. Each solution is of the form x-y-... where x is solved for and the optionally provided y-,... are deprojected. The x can belong to any of CMB,tSZ and y,z,... can belong to any of CMB,tSZ,CIB.')
parser.add_argument("beams", type=str,help='Comma separated list of beams. Each beam is either a float for FWHM in arcminutes or the name of an array whose beam will be used.')
parser.add_argument("-o", "--overwrite", action='store_true',help='Ignore existing version directory.')
parser.add_argument("-e", "--effective-freq", action='store_true',help='Ignore bandpass files and use effective frequency.')
parser.add_argument("--chunk-size",     type=int,  default=5000000,help="Chunk size.")
parser.add_argument("--maxval",     type=float,  default=700000,help="Maxval for covmat.")
parser.add_argument("--ccor-exp",     type=float,  default=-1,help="ccor exp.")
parser.add_argument("--pa1-shift",     type=float,  default=None,help="Shift of nu for pa1.")
parser.add_argument("--pa2-shift",     type=float,  default=None,help="Shift of nu for pa2.")
parser.add_argument("--pa3-150-shift",     type=float,  default=None,help="Shift of nu for pa3-150.")
parser.add_argument("--pa3-090-shift",     type=float,  default=None,help="Shift of nu for pa3-090.")
parser.add_argument("--beam-version", type=str,  default=None,help='Mask version')
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
