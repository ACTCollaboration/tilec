import healpy as hp
import numpy as np

"""
1. we start with maps of Nobs arrays ~ 20 (Planck and ACT) indexed by a,b
2. we apply an apodized mask to these
3. we define Ns~14 spectral windows indexed by j
4. we filter the maps with these to obtain Nobs*Ns ~  maps
5. for each j in Ns, we obtain Ma(x)*Mb(x) and smooth it with a Gaussian with width sigma_cov_j
6. We save these Ns*Nobs(Nobs+1)/2 product/covariance maps ~ 6510 maps

30,44,70,100,143,217,353,545 = 8
ar1,ar2,ar3,PA1,PA2,PA3a,PA3b,PA4a,PA4b,PA5a,PA5b,PA6a,PA6b = 13

How do we enforce ell space weighting?
"""


def bandlim_needlets(ells,lmins,lpeaks,lmaxs,tol=1e-8):
    filters = []
    ells = np.asarray(ells,dtype=np.float)
    lmins = np.asarray(lmins,dtype=np.float)
    lpeaks = np.asarray(lpeaks,dtype=np.float)
    lmaxs = np.asarray(lmaxs,dtype=np.float)
    for lmin,lpeak,lmax in zip(lmins,lpeaks,lmaxs):
        f = ells*0
        sel = np.logical_and(ells>=lmin,ells<=lpeak)
        f[sel] = np.cos( (lpeak-ells[sel]) / (lpeak-lmin) * np.pi / 2.)
        sel = np.logical_and(ells>lpeak,ells<=lmax)
        f[sel] = np.cos( (-lpeak+ells[sel]) / (lmax-lpeak) * np.pi / 2.)
        f[ells<2] = 0
        filters.append(f.copy())
    filters = np.asarray(filters)
    # assert (np.absolute( np.sum( filters**2., axis=0 ) - (ells*0 + 1)) < tol).all(), "wavelet filter transmission check failed"
    return filters

def gaussian_needlets(lmax,fwhm_arcmins=np.array([600., 300., 120., 60., 30., 15., 10., 7.5, 5.]),tol=1e-8):
    """
    Needlet spectral windows from J. Colin Hill
    """
    # Planck 2015 NILC y-map Gaussian needlet filters: [600', 300', 120', 60', 30', 15', 10', 7.5', 5']
    # Planck 2016 GNILC Gaussian needlet filters: [300' , 120' , 60' , 45' , 30' , 15' , 10' , 7.5' , 5']
    # (These are from email via M. Remazeilles 2/22/19 -- update: y-map filters are still slightly different at low ell than those in the paper)
    # for the details of the construction,
    #  see Eqs. (A.29)-(A.32) of http://arxiv.org/pdf/1605.09387.pdf
    # note that these can be constructed for different (user-specified) choices of N_scales and ELLMAX also
    # define the FWHM values used in the Gaussians -- default = Planck 2015 NILC y-map values
    # FWHM need to be in strictly decreasing order, otherwise you'll get nonsense
    if ( any( i <= j for i, j in zip(fwhm_arcmins, fwhm_arcmins[1:]))):
        raise AssertionError
    # check consistency with N_scales                                                                                  
    # assert(len(fwhm_arcmins) == self.N_scales - 1)
    N_scales = len(fwhm_arcmins) + 1
    FWHM = fwhm_arcmins * np.pi/(180.*60.)
    # define gaussians                                                                                          
    Gaussians = np.zeros((N_scales-1,lmax+1))
    for i in range(N_scales-1):
        Gaussians[i] = hp.gauss_beam(FWHM[i], lmax=lmax)
        # define needlet filters in harmonic space
    filters = np.ones((N_scales,lmax+1))
    filters[0] = Gaussians[0]
    for i in range(1,N_scales-1):
        filters[i] = np.sqrt(Gaussians[i]**2. - Gaussians[i-1]**2.)
    filters[N_scales-1] = np.sqrt(1. - Gaussians[N_scales-2]**2.)
    assert (np.absolute( np.sum( filters**2., axis=0 ) - np.ones(lmax+1,dtype=float)) < tol).all(), "wavelet filter transmission check failed"
    return filters
