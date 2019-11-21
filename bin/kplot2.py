from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap,reproject,enplot
import numpy as np
import os,sys
from soapack import interfaces as sints
import healpy as hp

"""
Plots filtered maps. Run kplot.py before.
"""

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("region", type=str,help='Region name.')
parser.add_argument("mtype", type=str,help='phi or def or kappa.')
parser.add_argument("--planck", action='store_true',help='Planck instead of ACT.')
args = parser.parse_args()

region = args.region
mtype = args.mtype
planck = args.planck



# Set contour level here
if mtype=='phi':
    nsigma = 0.5
elif mtype=='kappa':
    nsigma = 0.8
elif mtype=='def':
    nsigma = 0.8

if planck: pstr = "_planck"
else : pstr = ""

shape,wcs = enmap.read_map_geometry("fmap_%s_%s%s.fits" % (mtype,region,pstr))
Ny = shape[0] - 120
Nx = shape[1] - 120
fmap = maps.crop_center(enmap.read_map("fmap_%s_%s%s.fits"  % (mtype,region,pstr)),Ny,Nx)
hmap = maps.crop_center(enmap.read_map("hmap_%s_%s%s.fits"  % (mtype,region,pstr))[0],Ny,Nx)
p1 = enplot.plot(fmap,layers=True,color='gray',mask=0,ticks=2)
p1nol = enplot.plot(fmap,color='gray',mask=0,ticks=2)
enplot.write("%s_%s%s"  % (mtype,region,pstr), p1nol)

sig = hmap.std()
p2 = enplot.plot(hmap,layers=True,contours=nsigma*sig,contour_width=2,mask=0,contour_color='planck')

enplot.write("p2_%s_%s%s" % (mtype,region,pstr), p2)
p1 += [a for a in p2 if "cont" in a.name]
img = enplot.merge_images([a.img for a in p1])
enplot.write("combined_%s_%s%s.png"  % (mtype,region,pstr), img)
