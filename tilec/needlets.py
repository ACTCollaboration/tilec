import healpy as hp
import numpy as np
from pixell import curvedsky as cs
import h5py

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

class BandLimNeedlet(object):
    """
    A class for book-keeping of memory/disk efficient representation
    of needlet coefficient maps.
    """
    def __init__(self,ells,lmaxs,dir_path,debug_plots=False):
        lmins,lpeaks,self.filters = bandlim_needlets(ells,lmaxs)
        if debug_plots: plot_filters(ells,self.filters,dir_path,'linlin')
        self.ells = ells
        self.lmins = lmins
        self.lmaxs = lmaxs
        self.dpath = dir_path
        self.pcache = {}

    def get_part_alms(self,alms,ellmin,ellmax,mlmax):
        if mlmax in self.pcache.keys():
            ellarray = self.pcache[mlmax]
        else:
            ls = np.arange(mlmax+1.0,dtype=np.float32)
            ellarray = cs.almxfl(alms*0 + 1.0,fl=ls)
            self.pcache[mlmax] = ellarray
        sel = np.logical_and(ellarray>=ellmin,ellarray<ellmax)
        return alms[sel]

    def transform(self,tag,ellmin,ellmax,mlmax,imap=None,alm=None):
        if ellmin is None: ellmin = 0
        if ellmax is None: ellmax = np.inf
        assert ellmax>ellmin
        if alm is None: alm = cs.map2alm(imap,lmax=mlmax)
        fname = f'{self.dpath}/beta_alms_{tag}.h5'
        with h5py.File(fname, 'w') as f:
            for i,(flmin,flmax) in enumerate(zip(self.lmins,self.lmaxs)):
                outside = ((ellmax<flmin) or (ellmin>=flmax))
                if outside: continue
                beta_alm = self.get_part_alms(transform(self.filters[i][None],alm=alm)[0],flmin,flmax,mlmax)
                f.create_dataset(f'findex_{i}_flmin_{flmin}_flmax_{flmax}_ellmin_{ellmin}_ellmax_{ellmax}_mlmax_{mlmax}',data=beta_alm)
        

def transform(filters,imap=None,alm=None,lmax=None):
    """
    Given an (nscale,nells) filters array of needlet spectral windows,
    produces nscale filtered needlet coefficient maps. This really
    is just an almxfl operation.
    """
    if alm is None:
        alm = cs.map2alm(imap,lmax=lmax)
    betas = []
    for fl in filters:
        res = cs.almxfl(alm,fl=fl)
        if imap is not None: res = cs.alm2map(res,enmap.empty(imap.shape,imap.wcs,dtype=imap.dtype))
        betas.append(res)
    betas = np.asarray(betas)
    if imap is not None: betas = enmap.enmap(betas,imap.wcs)
    return betas
    

def bandlim_needlets(ells,lmaxs,tol=1e-8):
    filters = []
    ells = np.asarray(ells,dtype=np.float32)
    lmaxs = np.asarray(lmaxs,dtype=np.float32)
    lpeaks = np.append([0] , lmaxs[:-1])
    lmins = np.append([0], lpeaks[:-1])
    for lmin,lpeak,lmax in zip(lmins,lpeaks,lmaxs):
        assert lpeak>=lmin
        assert lmax>lpeak
        f = ells*0
        sel = np.logical_and(ells>=lmin,ells<lpeak)
        f[sel] = np.cos( (lpeak-ells[sel]) / (lpeak-lmin) * np.pi / 2.)
        f[np.isclose(ells,lpeak)] = 1.
        sel = np.logical_and(ells>lpeak,ells<lmax)
        f[sel] = np.cos( (-lpeak+ells[sel]) / (lmax-lpeak) * np.pi / 2.)
        f[ells<2] = 0
        filters.append(f.copy())
    filters = np.asarray(filters,dtype=np.float32)
    # assert (np.absolute( np.sum( filters**2., axis=0 ) - (ells*0 + 1)) < tol).all(), "wavelet filter transmission check failed"
    return lmins,lpeaks,filters

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


def plot_filters(ells,filters,dir_path,xyscale='linlin'):
    from orphics import io
    pl = io.Plotter(xyscale=xyscale,xlabel='l',ylabel='f')
    for i in range(filters.shape[0]): pl.add(ls[2:],filters[i,2:],label=str(i))
    trans = (filters[:,2:]**2.).sum(axis=0)
    print(ls[2:][trans<1-1e-5])
    pl.add(ls[2:],trans,color='k')
    pl.legend(loc='center left',bbox_to_anchor=(1,0.5))
    pl.done(f'{dir_path}filters.png')

