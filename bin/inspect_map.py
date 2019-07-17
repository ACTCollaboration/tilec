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
parser.add_argument("version", type=str,help='Region name.')
parser.add_argument("region", type=str,help='Region name.')
parser.add_argument("solution", type=str,help='Solution.')
parser.add_argument("--lmin",     type=int,  default=80,help="lmin.")
parser.add_argument("--lmax",     type=int,  default=6000,help="lmin.")
args = parser.parse_args()

save_path = sints.dconfig['tilec']['save_path']
savedir = save_path + args.version + "/" + args.region
name_map = {'CMB':'cmb','tSZ':'comptony','CIB':'cib'}
comps = "tilec_single_tile_"+args.region+"_" + name_map[args.solution]+"_"+args.version
lmin = args.lmin
lmax = args.lmax

imap = enmap.read_map("%s/%s.fits" % (savedir,comps))
modlmap = imap.modlmap()
kmap = enmap.fft(imap,normalize="phys")
p2d = np.real(kmap*kmap.conj())
N = 200
Ny,Nx = modlmap.shape
pimg = maps.crop_center(np.log10(np.fft.fftshift(p2d)),N,int(N*Nx/Ny))
if args.solution=='tSZ':
    lim = [-19,-15]
else:
    lim = [-5,1]
io.plot_img(pimg,"pimg.png",aspect='auto',lim=lim)
sel = np.logical_and(modlmap>lmin,modlmap<lmax)
xs = modlmap[sel].reshape(-1)
ys = p2d[sel].reshape(-1)

pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='C')
pl._ax.scatter(xs,ys)
#pl._ax.set_ylim(lim[0],lim[1])
pl.done("pscatter.png")

