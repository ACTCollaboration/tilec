from __future__ import print_function
from orphics import maps,io,cosmology
from sotools import enmap
import numpy as np
import os,sys
import tilec

f = lambda x : np.fft.fftshift(np.log10(x))

theory = cosmology.default_theory()
shape,wcs = maps.rect_geometry(width_deg = 5,px_res_arcmin=1)
modlmap = enmap.modlmap(shape,wcs)

p2d = enmap.enmap(theory.lCl('TT',modlmap),wcs)
bin_edges = np.arange(0.,modlmap.max(),40)
dp = tilec.signal_average(p2d,bin_edges)
io.plot_img(f(p2d))
io.plot_img(f(dp))
