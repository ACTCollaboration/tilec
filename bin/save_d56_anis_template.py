from tilec import utils as tutils,covtools
import numpy as np
from orphics import io
from pixell import enmap

"""
Calculates noise from splits.
Smooths it.
Saves the part with radial dependence divided out.
"""


config = tutils.Config(arrays = ["s15_pa3_150"])
_,ncov,_,npower = config.ncalc(0,0,do_coadd_noise=True)
ndown,nfitted,nparams = covtools.noise_average(npower,dfact=(16,16),lmin=300,lmax=8000,wnoise_annulus=500,bin_annulus=20,
                                               lknee_guess=3000,alpha_guess=-4,method="fft",radial_fit=True)

io.plot_img(tutils.tpower(ndown),"ndown.png",aspect='auto',lim=[-6,3])
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
