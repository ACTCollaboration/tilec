from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
import warnings
from enlib import bench


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
    with np.errstate(divide='ignore', invalid='ignore',over='ignore'):
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
    #ntemplatefunc = lambda x,lknee,alpha: fbinner.bin(rednoise(modlmap,wnoise,lknee=lknee,alpha=alpha))[1]
    ntemplatefunc = lambda x,lknee,alpha: rednoise(x,wnoise,lknee=lknee,alpha=alpha) # FIXME: This switch needs testing !!!!
    res,_ = curve_fit(ntemplatefunc,cents,dn1d,p0=[lknee_guess,alpha_guess])
    lknee_fit,alpha_fit = res

    # from orphics import io
    # pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='D',scalefn=lambda x: x**2./2./np.pi)
    # pl.add(cents,dn1d)
    # pl.add(cents,rednoise(cents,wnoise,lknee=lknee_fit,alpha=alpha_fit),ls="--")
    # pl.add(cents,rednoise(cents,wnoise,lknee=lknee_guess,alpha=alpha_guess),ls="-.")
    # pl.done("fitnoise.png")
    # sys.exit()

    return wnoise,lknee_fit,alpha_fit


def noise_average(n2d,dfact=(16,16),lmin=300,lmax=8000,wnoise_annulus=500,bin_annulus=20,
                  lknee_guess=3000,alpha_guess=-4,nparams=None,modlmap=None,
                  verbose=False,method="fft",radial_fit=True,
                  oshape=None,upsample=True,fill_lmax=None,fill_lmax_width=100):
    """Find the empirical mean noise binned in blocks of dfact[0] x dfact[1] . Preserves noise anisotropy.
    Most arguments are for the radial fitting part.
    A radial fit is divided out before downsampling (by default by FFT) and then multplied back with the radial fit.
    Watch for ringing in the final output.
    n2d noise power
    """
    assert np.all(np.isfinite(n2d))
    shape,wcs = n2d.shape,n2d.wcs
    minell = maps.minimum_ell(shape,wcs)
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
    if fill_lmax is not None:
        fill_avg = nflat[np.logical_and(modlmap>(fill_lmax-fill_lmax_width),modlmap<=fill_lmax)].mean()
        nflat[modlmap>fill_lmax] = fill_avg
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
    outcov = ndown*nfitted
    outcov[modlmap<minell] = 0 #np.inf
    if fill_lmax is not None: outcov[modlmap>fill_lmax] = 0



    # bad_ells = modlmap[np.isnan(outcov)]
    # for ell in bad_ells:
    #     print(ell)
    # print(maps.minimum_ell(shape,wcs))
    # from orphics import io
    # # io.hplot(enmap.enmap(np.fft.fftshift(np.log(n2d)),wcs),"fitnoise_npower.png")
    # # io.hplot(enmap.enmap(np.fft.fftshift(np.log(outcov)),wcs),"fitnoise_ndown.png")
    # io.plot_img(enmap.enmap(np.fft.fftshift(np.log(n2d)),wcs),"fitnoise_npower_lowres.png",aspect='auto')#,lim=[-20,-16])
    # io.plot_img(enmap.enmap(np.fft.fftshift(np.log(ndown)),wcs),"fitnoise_ndown_lowres.png",aspect='auto')#,lim=[-20,-16])
    # io.plot_img(enmap.enmap(np.fft.fftshift(np.log(outcov)),wcs),"fitnoise_outcov_lowres.png",aspect='auto')#,lim=[-20,-16])

    # import time
    # t = time.time()
    # fbin_edges = np.arange(lmin,lmax,bin_annulus)
    # fbinner = stats.bin2D(modlmap,fbin_edges)
    # cents, n1d = fbinner.bin(n2d)
    # cents,dn1d = fbinner.bin(outcov)
    # pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='D',scalefn=lambda x: x**2./2./np.pi)
    # pl.add(cents,n1d)
    # pl.add(cents,dn1d,ls="--")
    # pl.done("fitnoise2_%s.png" % t)
    # sys.exit()

    assert not(np.any(np.isnan(outcov)))
    return outcov,nfitted,nparams


log_smooth_corrections = [ 1.0, # dummy for 0 dof
 3.559160, 1.780533, 1.445805, 1.310360, 1.237424, 1.192256, 1.161176, 1.139016,
 1.121901, 1.109064, 1.098257, 1.089441, 1.082163, 1.075951, 1.070413, 1.065836,
 1.061805, 1.058152, 1.055077, 1.052162, 1.049591, 1.047138, 1.045077, 1.043166,
 1.041382, 1.039643, 1.038231, 1.036866, 1.035605, 1.034236, 1.033090, 1.032054,
 1.031080, 1.030153, 1.029221, 1.028458, 1.027655, 1.026869, 1.026136, 1.025518,
 1.024864, 1.024259, 1.023663, 1.023195, 1.022640, 1.022130, 1.021648, 1.021144,
 1.020772]

def smooth_ps_grid(ps, res, alpha=4, log=False, ndof=2):
    """Smooth a 2d power spectrum to the target resolution in l"""
    # First get our pixel size in l
    lx, ly = enmap.laxes(ps.shape, ps.wcs)
    ires   = np.array([lx[1],ly[1]])
    smooth = np.abs(res/ires)
    # We now know how many pixels to somoth by in each direction,
    # so perform the actual smoothing
    if log: ps = np.log(ps)
    fmap  = enmap.fft(ps)
    ky    = np.fft.fftfreq(ps.shape[-2])
    kx    = np.fft.fftfreq(ps.shape[-1])
    fmap /= 1 + np.abs(2*ky[:,None]*smooth[0])**alpha
    fmap /= 1 + np.abs(2*kx[None,:]*smooth[1])**alpha
    ps    = enmap.ifft(fmap).real
    if log: ps = np.exp(ps)*log_smooth_corrections[ndof]
    return ps


def noise_block_average(n2d,nsplits,delta_ell,lmin=300,lmax=8000,wnoise_annulus=500,bin_annulus=20,
                  lknee_guess=3000,alpha_guess=-4,nparams=None,
                  verbose=False,radial_fit=True,fill_lmax=None,fill_lmax_width=100):
    """Find the empirical mean noise binned in blocks of dfact[0] x dfact[1] . Preserves noise anisotropy.
    Most arguments are for the radial fitting part.
    A radial fit is divided out before downsampling (by default by FFT) and then multplied back with the radial fit.
    Watch for ringing in the final output.
    n2d noise power
    """
    assert np.all(np.isfinite(n2d))
    shape,wcs = n2d.shape,n2d.wcs
    modlmap = n2d.modlmap()
    minell = maps.minimum_ell(shape,wcs)
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
    nfitted = np.maximum(nfitted,np.max(n2d)*1e-14)
    nflat = enmap.enmap(n2d/nfitted,wcs) # flattened 2d noise power
    fval = nflat[np.logical_and(modlmap>2,modlmap<2*minell)].mean()
    nflat[modlmap<2] = fval
    if fill_lmax is not None:
        fill_avg = nflat[np.logical_and(modlmap>(fill_lmax-fill_lmax_width),modlmap<=fill_lmax)].mean()
        nflat[modlmap>fill_lmax] = fill_avg
    if verbose: print("Resampling...")
    assert np.all(np.isfinite(nflat))
    ndown = smooth_ps_grid(nflat, res=delta_ell, alpha=4, log=True, ndof=2*(nsplits-1))
    outcov = ndown*nfitted
    outcov[modlmap<minell] = 0
    if fill_lmax is not None: outcov[modlmap>fill_lmax] = 0
    assert np.all(np.isfinite(outcov))
    return outcov,nfitted,nparams


def signal_average(cov,bin_edges=None,bin_width=40,kind=3,lmin=None,dlspace=True,**kwargs):
    """
    dcov = cov * ellfact
    bin dcov in annuli
    interpolate back on to ell
    cov = dcov / ellfact
    where ellfact = ell**2 if dlspace else 1
    """
    assert np.all(np.isfinite(cov))
    modlmap = cov.modlmap()
    dcov = cov*modlmap**2. if dlspace else cov.copy()
    minell = maps.minimum_ell(dcov.shape,dcov.wcs) if lmin is None else lmin
    if bin_edges is None: bin_edges = np.arange(0,modlmap.max(),bin_width)
    binner = stats.bin2D(modlmap,bin_edges)
    cents,c1d = binner.bin(dcov)
    outcov = enmap.enmap(maps.interp(cents,c1d,kind=kind,**kwargs)(modlmap),dcov.wcs)
    outcov = outcov / modlmap**2. if dlspace else outcov
    outcov[modlmap<minell] = 0
    assert np.all(np.isfinite(outcov))
    return outcov



def get_anisotropic_noise_template(shape,wcs,template_file=None,tmin=0,tmax=100):
    """
    This function reads in a 2D PS unredenned template and returns a full 2D noise PS.
    It doesn't use the template in the most sensible way though.
    """
    if template_file is None: template_file = "data/anisotropy_template.fits"
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

