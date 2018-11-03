import os,sys
from tilec import utils as tutils,covtools
import numpy as np
from orphics import io,stats,cosmology,maps
from pixell import enmap

shape,wcs = maps.rect_geometry(width_deg=50.,height_deg=30,px_res_arcmin=0.5)
modlmap = enmap.modlmap(shape,wcs)
ells = np.arange(0,8000,1)
theory = cosmology.default_theory()
cltt2d = enmap.enmap(theory.lCl('TT',modlmap),wcs)
cltt2d[modlmap<50] = 0
cltt = theory.lCl('TT',ells)

ndown = covtools.signal_average(cltt2d,bin_width=40)

ny = int(shape[0]*5./100.)
nx = int(shape[1]*5./100.)

diff = maps.crop_center(np.fft.fftshift((ndown-cltt2d)/cltt2d),ny,nx)
io.plot_img(diff,"diff2d.png",aspect='auto',lim=[-0.1,0.1])
