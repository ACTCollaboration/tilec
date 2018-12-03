from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys


"""
TILe-C
ILC in tiles

We start with k_i splits each of N arrays labeled i in tile j
We assume a constant covariance model within the tile.
The total covariance is C = S + N, which we wish to estimate
within the tile j, where S is CMB+foregrounds and N is the
detector and atmospheric noise.

For simplicity, we assume that we have k_i=2 splits always.

We calculate Sab = <a_1b_2> , i.e. only the cross-split
spectra. The averaging is done in annular bins. This gives
us a coarse grained isotropic estimate of the signal covariance
S in tile j.

We next attempt to calculate the noise covariance without
losing anisotropy information. We do this by calculating
the 2D noise from the auto-spectrum (or in case of the 90-150
part the cross-spectrum) of difference maps. 
We obtain an intermediate 1D fit to the radial component 
which we divide out. We then downsample
this deconvolved 2D spectrum, and multiply the result by
the 1D radial fit that was divided out. This gives us
our coarse grained Nab estimate. 

We now have a total covariance Cab = Sab + Nab which we
can use for minimizing weights in ILC.

"""

def rednoise(ells,rms_noise,lknee=0.,alpha=1.):
    """Atmospheric noise model
    rms_noise in muK-arcmin
    [(lknee/ells)^(-alpha) + 1] * rms_noise**2
    """
    atm_factor = (lknee*np.nan_to_num(1./ells))**(-alpha) if lknee>1.e-3 else 0.
    rms = rms_noise * (1./60.)*(np.pi/180.)
    wnoise = ells*0.+rms**2.
    return (atm_factor+1.)*wnoise


def fit_noise_1d(npower,lmin=300,lmax=10000,wnoise_annulus=500,bin_annulus=20,lknee_guess=3000,alpha_guess=-4):
    """Obtain a white noise + lknee + alpha fit to a 2D noise power spectrum
    The white noise part is inferred from the mean of lmax-wnoise_annulus < ells < lmax
    
    npower is 2d noise power
    """
    from scipy.optimize import curve_fit
    fbin_edges = np.arange(lmin,lmax,bin_annulus)
    modlmap = npower.modlmap()
    fbinner = stats.bin2D(modlmap,fbin_edges)
    cents,dn1d = fbinner.bin(npower)
    wnoise = np.sqrt(dn1d[np.logical_and(cents>=(lmax-wnoise_annulus),cents<lmax)].mean())*180.*60./np.pi
    ntemplatefunc = lambda x,lknee,alpha: fbinner.bin(rednoise(modlmap,wnoise,lknee=lknee,alpha=alpha))[1]
    res,_ = curve_fit(ntemplatefunc,cents,dn1d,p0=[lknee_guess,alpha_guess])
    lknee_fit,alpha_fit = res
    return wnoise,lknee_fit,alpha_fit


def noise_average(n2d,dfact=(16,16),lmin=300,lmax=8000,wnoise_annulus=500,bin_annulus=20,
                  lknee_guess=3000,alpha_guess=-4,nparams=None,modlmap=None,
                  verbose=False,method="fft",radial_fit=True,
                  oshape=None,upsample=True):
    """Find the empirical mean noise binned in blocks of dfact[0] x dfact[1] . Preserves noise anisotropy.
    Most arguments are for the radial fitting part.
    A radial fit is divided out before downsampling (by default by FFT) and then multplied back with the radial fit.
    Watch for ringing in the final output.
    n2d noise power
    """
    shape,wcs = n2d.shape,n2d.wcs
    if modlmap is None: modlmap = enmap.modlmap(shape,wcs)
    Ny,Nx = shape[-2:]
    if radial_fit:
        if nparams is None:
            if verbose: print("Radial fitting...")
            nparams = fit_noise_1d(n2d,lmin=lmin,lmax=lmax,wnoise_annulus=wnoise_annulus,
                                bin_annulus=bin_annulus,lknee_guess=lknee_guess,alpha_guess=alpha_guess)
        wfit,lfit,afit = nparams
        nfitted = rednoise(modlmap,wfit,lfit,afit)
    else:
        nparams = None
        nfitted = 1.
    nflat = enmap.enmap(np.nan_to_num(n2d/nfitted),wcs) # flattened 2d noise power
    if oshape is None: oshape = (Ny//dfact[0],Nx//dfact[1])
    if verbose: print("Resampling...")
    nint = enmap.resample(enmap.enmap(nflat,wcs), oshape, method=method)
    if not(upsample):
        if radial_fit:
            nshape,nwcs = nint.shape,nint.wcs
            modlmap = enmap.modlmap(nshape,nwcs)
            nfitted = rednoise(modlmap,wfit,lfit,afit)
        ndown = nint
    else:
        ndown = enmap.enmap(enmap.resample(nint,shape,method=method),wcs)
    return ndown*nfitted,nfitted,nparams


def signal_average(cov,bin_edges=None,bin_width=40,kind=5,**kwargs):
    modlmap = cov.modlmap()
    if bin_edges is None: bin_edges = np.arange(0,modlmap.max(),bin_width)
    binner = stats.bin2D(modlmap,bin_edges)
    cents,c1d = binner.bin(cov)
    outcov = maps.interp(cents,c1d,kind=kind,**kwargs)(modlmap)
    return outcov



def get_anisotropic_noise_template(shape,wcs,template_file=None,tmin=0,tmax=100):
    """
    This function reads in a 2D PS unredenned template and returns a full 2D noise PS.
    It doesn't use the template in the most sensible way though.
    """
    if template_file is None:
        template_file = "data/anisotropy_template.fits"
    template = np.nan_to_num(enmap.read_map(template_file))
    template[template<tmin] = tmin
    template[template>tmax] = tmax
    ops = enmap.enmap(enmap.resample(template,shape),wcs) # interpolate to new geometry
    return ops

def get_anisotropic_noise(shape,wcs,rms,lknee,alpha,template_file=None,tmin=0,tmax=100):
    """
    This function reads in a 2D PS unredenned template and returns a full 2D noise PS.
    It doesn't use the template in the most sensible way though.
    """
    ops = get_anisotropic_noise_template(shape,wcs,template_file,tmin,tmax)
    return rednoise(enmap.modlmap(shape,wcs),rms,lknee,alpha)*ops
