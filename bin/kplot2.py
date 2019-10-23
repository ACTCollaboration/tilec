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


# Toggle Planck here
planck = False
# Toggle region here
#region = 'boss'
region = 'deep56'


# Set contour level here
nsigma = 0.5

if planck: pstr = "_planck"
else : pstr = ""

shape,wcs = enmap.read_map_geometry("fmap_%s%s.fits" % (region,pstr))
Ny = shape[0] - 120
Nx = shape[1] - 120
fmap = maps.crop_center(enmap.read_map("fmap_%s%s.fits"  % (region,pstr)),Ny,Nx)
hmap = maps.crop_center(enmap.read_map("hmap_%s%s.fits"  % (region,pstr))[0],Ny,Nx)
p1 = enplot.plot(fmap,layers=True,color='gray',mask=0,ticks=2)
p1nol = enplot.plot(fmap,color='gray',mask=0,ticks=2)
enplot.write("phi_%s%s"  % (region,pstr), p1nol)

sig = hmap.std()
p2 = enplot.plot(hmap,layers=True,contours=nsigma*sig,contour_width=2,mask=0,contour_color='planck')

enplot.write("p2_%s%s" % (region,pstr), p2)
p1 += [a for a in p2 if "cont" in a.name]
img = enplot.merge_images([a.img for a in p1])
enplot.write("combined_%s%s.png"  % (region,pstr), img)
