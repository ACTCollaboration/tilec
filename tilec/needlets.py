import healpy as hp
import numpy as np
from pixell import curvedsky as cs, enmap, utils,bunch
import h5py
import pandas as pd
from enlib import bench
from orphics import maps

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

def get_needlet_map_geometries(dm,df_bounds,qids,pxres,lmins,lmaxs,mask_geometries):
    geometries = {}
    bounds = {}
    for qid in qids:
        ellmin,ellmax,mlmax = get_bounds(dm,qid,df_bounds)
        bounds[qid] = bunch.Bunch({})
        bounds[qid].ellmin = ellmin
        bounds[qid].ellmax = ellmax
        bounds[qid].mlmax = mlmax

    for i,(flmin,flmax,px) in enumerate(zip(lmins,lmaxs,pxres)):
        geometries[i] = {}
        for qid in qids:
            ellmin,ellmax,mlmax = bounds[qid].ellmin,bounds[qid].ellmax,bounds[qid].mlmax
            outside = ((ellmax<flmin) or (ellmin>=flmax))
            if outside: continue
            
            bshape,bwcs = enmap.fullsky_geometry(res=px * utils.arcmin)
            oshape,owcs = mask_geometries[qid]
            box = enmap.box(oshape,owcs)
            # FIXME: This is really clunky, ugly and unnecessary, but I couldn't find a better way
            # that doesn't require writing a new enmap function
            omap = enmap.empty(bshape,bwcs,dtype=np.int).submap(box)
            
            geometries[i][qid] = bunch.Bunch({})
            geometries[i][qid].shape = omap.shape
            geometries[i][qid].wcs = omap.wcs
    return geometries,bounds

def get_bounds(dm,qid,df_bounds):
    adf = df_bounds
    if qid not in dm.planck_qids: qid = 'act'
    ellmin = adf[adf['#qid']==qid]['ellmin'].iloc[0]
    ellmax = adf[adf['#qid']==qid]['ellmax'].iloc[0]
    mlmax = adf[adf['#qid']==qid]['mlmax'].iloc[0]
    return ellmin,ellmax,mlmax


class BandLimNeedlet(object):
    """
    A class for book-keeping of memory/disk efficient representation
    of needlet coefficient maps.
    """
    def __init__(self,mode,qids,dm,dir_path,mask_geometries,debug_plots=False):
        lmaxs,pxres = np.loadtxt(f"data/needlet_lmaxs_{mode}.txt",delimiter=',',unpack=True)
        ells = np.arange(lmaxs.max()+1)
        df_bounds = pd.read_csv(f"data/needlet_bounds_{mode}.txt")
        lmins,lpeaks,self.filters = bandlim_needlets(ells,lmaxs)
        self.geometries,self.bounds = get_needlet_map_geometries(dm,df_bounds,qids,pxres,lmins,lmaxs,mask_geometries)
        print(self.geometries[0].keys())
        if debug_plots: plot_filters(ells,self.filters,dir_path,'linlin')
        self.dm = dm
        self.ells = ells
        self.lmins = lmins
        self.lmaxs = lmaxs
        self.dpath = dir_path
        self.pxres = pxres
        self.pcache = {}
        self.nfilters = self.filters.shape[0]
        assert len(lmins) == len(lmaxs) == self.nfilters
        self.fqids = [list(self.geometries[i].keys()) for i in range(self.nfilters)]

    def transform(self,qid,tag,imap=None,alm=None,forward=True,target_fwhm_arcmin=None):
        ellmin,ellmax,mlmax = self.bounds[qid].ellmin, self.bounds[qid].ellmax, self.bounds[qid].mlmax
        if ellmin is None: ellmin = 0
        if ellmax is None: ellmax = np.inf
        assert ellmax>ellmin
        if alm is None: alm = cs.map2alm(imap,lmax=mlmax)
        if forward:
            # beam_recon_filt = 1 # FIXME: turn this off !!!!
            # Only the forward transform does a beam reconvolution
            beam_fn = self.dm.get_beam_func(qid,sanitize=True)
            beam_recon_filt = maps.gauss_beam(self.ells,target_fwhm_arcmin)/beam_fn(self.ells)
            beam_recon_filt[self.ells<2] = 0
        else:
            beam_recon_filt = 1

        fname = f'{self.dpath}/beta_maps_{qid}_{tag}.h5'
        with h5py.File(fname, 'w') as f:
            for i,(flmin,flmax) in enumerate(zip(self.lmins,self.lmaxs)):
                outside = ((ellmax<flmin) or (ellmin>=flmax))
                if outside: continue
                fl = self.filters[i]*beam_recon_filt
                fl[self.ells>=flmax] = 0
                beta_alm = cs.almxfl(alm,fl=fl)
                shape,wcs = self.geometries[i][qid].shape, self.geometries[i][qid].wcs
                beta = cs.alm2map(beta_alm,enmap.empty(shape,wcs,dtype=np.float32))
                f.create_dataset(f'findex_{i}',data=beta)
        
    def load_beta(self,findex,qid,tag):
        fname = f'{self.dpath}/beta_maps_{qid}_{tag}.h5'
        with h5py.File(fname, 'r') as f:
            data = f[f'findex_{findex}'][:]
        shape,wcs = self.geometries[findex][qid].shape, self.geometries[findex][qid].wcs
        assert np.all(data.shape==shape)
        return enmap.enmap(data,wcs)
    

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
    ls = ells
    from orphics import io
    pl = io.Plotter(xyscale=xyscale,xlabel='l',ylabel='f')
    for i in range(filters.shape[0]): pl.add(ls[2:],filters[i,2:],label=str(i))
    trans = (filters[:,2:]**2.).sum(axis=0)
    print(ls[2:][trans<1-1e-5])
    pl.add(ls[2:],trans,color='k')
    pl.legend(loc='center left',bbox_to_anchor=(1,0.5))
    pl.done(f'{dir_path}filters.png')

