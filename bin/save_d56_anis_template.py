from tilec import utils as tutils,covtools
import numpy as np
from orphics import io,maps
from pixell import enmap
from soapack import interfaces as sints
import tilec.fg as tfg
import tilec.utils as tutils
from actsims import noise
from tilec import covtools

"""
Calculates noise from splits.
Smooths it.
Saves the part with radial dependence divided out.
"""


# config = tutils.Config(arrays = ["s15_pa3_150"])
# _,ncov,_,npower = config.ncalc(0,0,do_coadd_noise=True)


region = 'deep56'
season = 's15'
patch = region
array = 'pa3'

mask = sints.get_act_mr3_crosslinked_mask(region)

dm = sints.models['act_mr3'](region=mask,calibrated=True)
splits = dm.get_splits(season=season,patch=patch,arrays=dm.array_freqs[array],srcfree=True)
ivars = dm.get_splits_ivar(season=season,patch=patch,arrays=dm.array_freqs[array])

npower = noise.get_n2d_data(splits,ivars,mask,coadd_estimator=True,
                               flattened=False,
                               plot_fname=None,
                               dtype=dm.dtype)


ndown,nfitted,nparams = covtools.noise_average(npower[0,0],dfact=(16,16),lmin=300,lmax=8000,wnoise_annulus=500,bin_annulus=20,
                                               lknee_guess=3000,alpha_guess=-4,method="fft",radial_fit=True)

#io.plot_img(np.fft.fftshift(np.log10(ndown)),"ndown.png",aspect='auto',lim=[-6,3])
#io.hplot(np.fft.fftshift(np.log10(ndown)),"hndown")
io.hplot(np.fft.fftshift((ndown)),"hndown")
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
