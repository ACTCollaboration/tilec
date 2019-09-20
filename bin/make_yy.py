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
parser.add_argument("--unblind", action='store_true',help='Unblind the spectra.')
parser.add_argument("--all-analytic", action='store_true',help='All covs analytic.')
parser.add_argument("--no-samples", action='store_true',help='All covs analytic.')

args = parser.parse_args()

from szar import foregrounds as fgs

print("Command line arguments are %s." % args)

bin_edges = np.arange(600,4000,100)
cov_versions = ['yy_split_0_v1.0.0_rc','yy_split_1_v1.0.0_rc','v1.0.0_rc']
bsamples = np.random.normal(loc=1.2,scale=0.2,size=40) if not(args.no_samples) else None

pells,pyy = np.loadtxt("data/planck_yy.csv",delimiter=',',unpack=True)
pyy = pyy/pells/(pells+1)*2*np.pi

cents,powers,ysamples,cysamples = pipeline.calculate_yy(bin_edges,args.arrays,args.region,args.version,cov_versions,args.beam_version,
                                     args.effective_freq,args.overwrite,args.maxval,unsanitized_beam=args.unsanitized_beam,do_weights=True,
                                     pa1_shift = args.pa1_shift,
                                     pa2_shift = args.pa2_shift,
                                     pa3_150_shift = args.pa3_150_shift,
                                     pa3_090_shift = args.pa3_090_shift,
                                     no_act_color_correction=args.no_act_color_correction,ccor_exp=args.ccor_exp,
                                     sim_splits=None,unblind=args.unblind,all_analytic=args.all_analytic,beta_samples = bsamples)


pl = io.Plotter(xlabel='l',ylabel='D',scalefn = lambda x: x**2./2./np.pi,xyscale='linlog')
ells = np.arange(100,6000,1)
cyy = fgs.power_y(ells)
pl.add(ells,cyy,color='k')
for key in powers.keys():
    pl.add(cents,powers[key],label=key,ls="--")
if not(args.no_samples):
    for i in range(len(bsamples)):
        pl.add(cents,ysamples[i],color='blue',alpha=0.2)
        pl.add(cents,cysamples[i],color='red',alpha=0.2)

pl.add(pells,pyy,ls="none",marker="o",color='k')
pl._ax.set_ylim(1e-13,1e-10)
pl.done(os.environ['WORK'] + "/yy.png")
