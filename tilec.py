from __future__ import print_function
from orphics import maps,io,cosmology,stats
from sotools import enmap
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
    '''
    rms_noise in muK-arcmin
    [(lknee/ells)^(-alpha) + 1] * rms_noise**2
    '''
    atm_factor = (lknee*np.nan_to_num(1./ells))**(-alpha) if lknee>1.e-3 else 0.
    rms = rms_noise * (1./60.)*(np.pi/180.)
    wnoise = ells*0.+rms**2.
    return (atm_factor+1.)*wnoise


def fit_noise_1d(npower,lmin=300,lmax=10000,wnoise_annulus=500,bin_annulus=20,lknee_guess=3000,alpha_guess=-4):
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
                  lknee_guess=3000,alpha_guess=-4,nparams=None,modlmap=None,verbose=False,method="fft"):
    shape,wcs = n2d.shape,n2d.wcs
    if modlmap is None: modlmap = enmap.modlmap(shape,wcs)
    Ny,Nx = shape[-2:]
    if nparams is None:
        if verbose: print("Radial fitting...")
        nparams = fit_noise_1d(n2d,lmin=lmin,lmax=lmax,wnoise_annulus=wnoise_annulus,
                            bin_annulus=bin_annulus,lknee_guess=lknee_guess,alpha_guess=alpha_guess)
    wfit,lfit,afit = nparams
    nfitted = rednoise(modlmap,wfit,lfit,afit)
    nflat = enmap.enmap(np.nan_to_num(n2d/nfitted),wcs)
    oshape = (Ny//dfact[0],Nx//dfact[1])
    if verbose: print("Resampling...")
    nint = enmap.resample(enmap.enmap(np.fft.fftshift(nflat),wcs), oshape, method=method)
    ndown = enmap.enmap(np.fft.ifftshift(enmap.resample(nint,shape,method=method)),wcs)
    return ndown*nfitted,nparams


def noise_from_splits(splits,fourier_calc=None,nthread=0,do_cross=True):
    """
    Calculate noise power spectra by subtracting cross power of splits 
    from autopower of splits. Optionally calculate cross power spectra
    of T,E,B from I,Q,U.

    splits -- (nsplits,ncomp,Ny,Nx) arrays

    ncomp can be 1 for T only, or 3 for I,Q,U
    ncomp could be > 3 for e.g. I1,Q1,U1,I2,Q2,U2 for 2 arrays

    """

    try:
        wcs = splits.wcs
    except:
        wcs = splits[0].wcs
        
    splits = enmap.enmap(np.asarray(splits),wcs).astype(np.float32)
    assert splits.ndim==3 or splits.ndim==4
    if splits.ndim == 3: splits = splits[:,None,:,:]
    ncomp = splits.shape[1]
    ndim = splits.ndim
        
    if fourier_calc is None:
        shape = splits.shape[-3:] if do_cross else splits.shape[-2:]
        fourier_calc = FourierCalc(shape,wcs)
    
    Nsplits = splits.shape[0]

    if do_cross: assert ncomp==3 or ncomp==1


    # Get fourier transforms of I,Q,U
    ksplits = [fourier_calc.iqu2teb(split, nthread=nthread, normalize=False, rot=False) for split in splits]
    del splits
    
    if do_cross:
        kteb_splits = []
        # Rotate I,Q,U to T,E,B for cross power (not necssary for noise)
        for ksplit in ksplits:
            kteb_splits.append( ksplit.copy())
            if (ndim==3 and ncomp==3):
                kteb_splits[-1][...,-2:,:,:] = enmap.map_mul(fourier_calc.rot, kteb_splits[-1][...,-2:,:,:])
            
    # get auto power of I,Q,U
    auto = 0.
    for ksplit in ksplits:
        auto += fourier_calc.power2d(kmap=ksplit)[0]
    auto /= Nsplits

    # do cross powers of I,Q,U
    Ncrosses = (Nsplits*(Nsplits-1)/2)
    cross = 0.
    for i in range(len(ksplits)):
        for j in range(i+1,len(ksplits)):
            cross += fourier_calc.power2d(kmap=ksplits[i],kmap2=ksplits[j])[0]
    cross /= Ncrosses
        
    if do_cross:
        # do cross powers of T,E,B
        cross_teb = 0.
        for i in range(len(ksplits)):
            for j in range(i+1,len(ksplits)):
                cross_teb += fourier_calc.power2d(kmap=kteb_splits[i],kmap2=kteb_splits[j])[0]
        cross_teb /= Ncrosses
    else:
        cross_teb = None
    del ksplits

    # get noise model for I,Q,U
    noise = (auto-cross)/Nsplits

    # return I,Q,U noise model and T,E,B cross-power
    return noise,cross_teb


def fit_radial(power2d,lknee_guess,alpha_guess,wnoise):
    pass

def get_noise_cov(imaps):
    """
    imaps - (nsplits,narrays,Ny,Nx)
    """

    npow = maps.noise_from_splits(imaps,fourier_calc=None,nthread=0,do_cross=False)

    
