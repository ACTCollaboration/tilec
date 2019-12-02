from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from orphics import maps,io,cosmology,stats
from pixell import enmap,curvedsky as cs,enplot
import numpy as np
import os,sys
from soapack import interfaces as sints
from tilec import utils as tutils
import healpy as hp

import numpy as np
from pixell import enmap, utils, enplot
m1 = enmap.read_map("/scratch/r/rbond/msyriac/data/for_sigurd/planck_353.fits")
m2 = enmap.read_map("/scratch/r/rbond/msyriac/data/for_sigurd/mask_boss.fits", geometry=m1.geometry) + enmap.read_map("/scratch/r/rbond/msyriac/data/for_sigurd/mask_deep56.fits", geometry=m1.geometry)

print(m1.wcs)
print(m2.wcs)

off = [0,1000]
m1s = enmap.shift(enmap.downgrade(m1,10), off)
m2s = enmap.shift(enmap.downgrade(m2,10), off)

p1 = enplot.plot(m1s, min=500, max=1e4, color="cooltowarm", ticks="10", layers=True,font_size=50,contour_color='ffffff')
p2 = enplot.plot(m2s, min=0, max=1, ticks="10", layers=True, contours="0.5,", contour_width=6, annotate="paper/annot.txt",font_size=50,contour_color='ffffff') # yes, yes, I know

ptot = enplot.merge_plots(p1 + p2[1:3])
enplot.write("fig_footprint.png", ptot)
