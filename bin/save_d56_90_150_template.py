import os,sys
from tilec import utils as tutils,covtools
import numpy as np
from orphics import io,stats
from pixell import enmap

"""
Calculates noise from splits.
Smooths it.
Saves the part with radial dependence divided out.
"""


config = tutils.Config(arrays = ["s15_pa3_150","s15_pa3_90"])
# _,ncov,_,npower00 = config.ncalc(0,0,do_coadd_noise=True)
# _,ncov,_,npower11 = config.ncalc(1,1,do_coadd_noise=True)
# _,ncov,_,npower01 = config.ncalc(0,1,do_coadd_noise=True)

# npower = npower01
# io.plot_img(tutils.tpower(npower),"npower.png",aspect='auto',lim=[-6,3])

# sys.exit()
npower = enmap.read_map("npower.fits")
shape,wcs = npower.shape,npower.wcs
modlmap = enmap.modlmap(shape,wcs)
# wnoise,lknee_fit,alpha_fit = covtools.fit_noise_1d(npower,lmin=300,lmax=10000,
#                                                    wnoise_annulus=500,bin_annulus=20,
#                                                    lknee_guess=3000,alpha_guess=-4)
# fit2d = covtools.rednoise(modlmap,wnoise,lknee=lknee_fit,alpha=alpha_fit)
bin_edges = np.arange(100,8000,40)
binner = stats.bin2D(modlmap,bin_edges)

ndown,nfitted,nparams = covtools.noise_average(npower,dfact=(16,16),lmin=100,lmax=8000,wnoise_annulus=500,bin_annulus=20,
                                               lknee_guess=3000,alpha_guess=-4,method="fft",radial_fit=True)
ndown2 = covtools.signal_average(npower,bin_width=40)
io.plot_img(tutils.tpower(ndown),"ndown.png",aspect='auto')#,lim=[-6,3])

cents,n1d = binner.bin(ndown)
cents,n1d0 = binner.bin(npower)
cents,d1d = binner.bin(np.nan_to_num((npower-ndown)/ndown))
cents,d1d2 = binner.bin(np.nan_to_num((npower-ndown2)/ndown2))
# cents,f1d = binner.bin(fit2d)
pl = io.Plotter(xlabel='l',ylabel='C',yscale='log')
pl.add(cents,n1d0)
pl.add(cents,n1d,ls="--",lw=3)
# pl.add(cents,f1d,ls="--")
pl.done("n1d.png")
pl = io.Plotter(xlabel='l',ylabel='D')
pl.add(cents,d1d)
pl.add(cents,d1d2,ls="--")
pl.hline()
pl._ax.set_ylim(-0.2,0.2)
pl.done("d1d.png")

sys.exit()

#ndown,nfitted,nparams = covtools.noise_average(npower,dfact=(16,16),lmin=300,lmax=8000,wnoise_annulus=500,bin_annulus=20,
#                                               lknee_guess=3000,alpha_guess=-4,method="fft",radial_fit=False)
ndown = covtools.signal_average(npower,bin_width=40)
io.plot_img(tutils.tpower(ndown),"ndown.png",aspect='auto')#,lim=[-6,3])
sys.exit()
io.plot_img(np.fft.fftshift(ndown/nfitted),"nunred.png",aspect='auto')

nmod = ndown/nfitted
enmap.write_map("anisotropy_template.fits",enmap.samewcs(nmod,npower))


shape,wcs = maps.rect_geometry(width_deg=50.,height_deg=30,px_res_arcmin=0.5)
rms = 10.0
lknee = 3000
alpha = -3
n2d = covtools.get_anisotropic_noise(shape,wcs,rms,lknee,alpha)
modlmap = enmap.modlmap(shape,wcs)
bin_edges = np.arange(100,8000,100)
binner = stats.bin2D(modlmap,bin_edges)
cents,n1d = binner.bin(n2d)

pl = io.Plotter(yscale='log',xlabel='l',ylabel='C')
pl.add(cents,n1d)
pl.add(cents,covtools.rednoise(cents,rms,lknee=lknee,alpha=alpha),ls="--")
pl.done()
