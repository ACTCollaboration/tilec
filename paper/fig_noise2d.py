from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
from orphics import maps,io,cosmology,catalogs,stats
from pixell import enmap,reproject
import numpy as np
import os,sys
from soapack import interfaces as sints
import tilec.fg as tfg
import tilec.utils as tutils
from actsims import noise
from tilec import covtools

region = 'deep56'
season = 's15'
patch = region
array = 'pa3'

mask = sints.get_act_mr3_crosslinked_mask(region)

dm = sints.models['act_mr3'](region=mask,calibrated=True)
splits = dm.get_splits(season=season,patch=patch,arrays=dm.array_freqs[array],srcfree=True)
ivars = dm.get_splits_ivar(season=season,patch=patch,arrays=dm.array_freqs[array])

n2d_xflat = noise.get_n2d_data(splits,ivars,mask,coadd_estimator=True,
                               flattened=False,
                               plot_fname=None,
                               dtype=dm.dtype)
modlmap = splits.modlmap()
bin_edges = np.arange(20,8000,20)
binner = stats.bin2D(modlmap,bin_edges)
n2d = n2d_xflat[3,3]
cents,n1d = binner.bin(n2d)
pl = io.Plotter(xyscale='linlog',scalefn = lambda x: x**2,xlabel='l',ylabel='D*2pi')
pl.add(cents,n1d)
pl.done('n1d.png')
nsplits = 4
delta_ell = 400

n2d_xflat_smoothed,_,_ = covtools.noise_block_average(n2d,nsplits,delta_ell,lmin=300,lmax=8000,wnoise_annulus=500,bin_annulus=20,
                                                      lknee_guess=3000,alpha_guess=-4,nparams=None,log=True,radial_fit=False)

print(n2d_xflat.shape,n2d_xflat_smoothed.shape)
N = 1200
Ny,Nx = n2d_xflat_smoothed.shape[-2:]
M = maps.crop_center(np.fft.fftshift(modlmap),N,int(N*Nx/Ny))
d = maps.crop_center(np.fft.fftshift(n2d_xflat_smoothed),N,int(N*Nx/Ny))


# io.hplot(np.log10(d),'fig_hnoise',colorbar=True)
# io.plot_img(np.log10(np.fft.fftshift(n2d_xflat_smoothed)),'fig_tlognoise.png',aspect='auto')
io.plot_img(maps.crop_center(np.log10(np.fft.fftshift(n2d_xflat_smoothed)),N,int(N*Nx/Ny)),"fig_noise.pdf" ,aspect='auto',xlabel='$\\ell_x$',ylabel='$\\ell_y$',arc_width=2*M[0,0],lim=[-4.39,-3.46],label="$\\rm{log}_{10}(N ~\\mu{\\rm K}^2\\cdot {\\rm sr})$")
