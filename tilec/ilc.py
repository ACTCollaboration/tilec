import numpy as np
from pixell import utils
from tilec import covtools

"""
This module implements harmonic ILC.
"""

def cinv_x(x,cov,cinv):
    """Dot Cinv with x, either with the provided inverse, or with linalg.solve"""
    if cinv is None: Cinvx = np.linalg.solve(cov,x)
    else: Cinvx = np.einsum('...ij,...j->...i',cinv,x)
    return Cinvx

def map_comb(response_a,response_b,cov=None,cinv=None):
    """Return a^T Cinv b"""
    # Cinv b = np.linalg.solve(cov,b)
    # Cov is in shape (...n,n)
    Cinvb = cinv_x(response_b,cov,cinv)
    return np.nan_to_num(np.einsum('...l,...l->...',response_a,Cinvb))

def map_term(kmaps,response,cov=None,cinv=None):
    """response^T . Cinv . kmaps """
    Cinvk = cinv_x(kmaps,cov,cinv)
    return np.einsum('...k,...k->...',response,Cinvk)

class HILC(object):
    """
    Harmonic ILC.
    We avoid beam deconvolution, instead modeling the beam in the response.
    Since all maps are beam convolved, we do not need lmaxes.
    """
    def __init__(self,ells,kbeams,cov=None,responses=None,chunks=1,invert=True):
        """
        Args:
            ells: (nells,) or (Ny,Nx) specifying mode number mapping for each pixel
            kbeams: (nmap,nells) or (nmap,Ny,Nx) fourier space beam factor. nmap
            determines number of frequencies/arrays used.
            cov: (nmap,nmap,nells) or (nmap,nmap,Ny,Nx) covariance matrix of 
            beam-convolved maps.
            responses: dictionary mapping component name to (nmap,) floats specifying
            the frequency/array response to that component for a beam-deconvolved map.
            chunks: integer factor by which to divide the ell space into and loop
            through to conserve memory
        """
        # Unravel ells and beams
        nmap = kbeams.shape[0]
        if ells.ndim==2:
            self._2d = True
            ells = ells.reshape(-1)
            assert kbeams.ndim==3
            kbeams = kbeams.reshape((nmap,ells.size))
            cov = cov.reshape((nmap,nmap,ells.size))
        elif ells.ndim==1:
            self._2d = False
            assert kbeams.ndim==2
        else: raise ValueError
        kbeams = kbeams.swapaxes(0,1)
        self.ells = ells
        self.nmap = nmap
        self.chunks = chunks
        self.kbeams = kbeams
        self.cov = np.moveaxis(cov,(0,1),(-2,-1))
        if invert:
            self.cinv = np.linalg.inv(self.cov) #utils.eigpow(np.nan_to_num(self.cov),-1,alim=0,rlim=0)
        else: self.cinv = None
        self.responses = {}
        if responses is None: responses = {}
        if "cmb" not in responses.keys(): responses['cmb'] = np.ones((1,nmap))
        for key in responses.keys():
            self.add_response(key,responses[key].reshape((1,nmap)))
                
    def add_response(self,name,response):
        self.responses[name] = response * self.kbeams

    def cross_noise(self,name1,name2):
        """
        Cross-noise of <standard constrained>
        """
        response_a = self.responses[name1]
        response_b = self.responses[name2]
        return cross_noise(response_a,response_b,self.cov,self.cinv)
        
    def standard_noise(self,name):
        """
        Auto-noise <standard standard>
        """
        r = self.responses[name]
        return standard_noise(r,self.cov,self.cinv)

    def constrained_noise(self,name1,name2,return_cross=False):
        """ Derived from Eq 18 of arXiv:1006.5599
        Auto-noise <constrained constrained>
        """
        response_a = self.responses[name1]
        response_b = self.responses[name2]
        return constrained_noise(response_a,response_b,self.cov,self.cinv,return_cross)

    def _prepare_maps(self,kmaps):
        assert kmaps.shape[0] == self.nmap
        if self._2d: kmaps = kmaps.reshape((self.nmap,self.ells.size))
        kmaps = kmaps.swapaxes(0,1)
        return kmaps
        
    def standard_map(self,kmaps,name="cmb"):
        # Get response^T cinv kmaps
        kmaps = self._prepare_maps(kmaps)
        weighted = map_term(kmaps,self.responses[name],self.cov,self.cinv)
        return weighted * self.standard_noise(name)

    def constrained_map(self,kmaps,name1,name2):
        """Constrained ILC -- Make a constrained internal linear combination (ILC) of given fourier space maps at different frequencies
        and an inverse covariance matrix for its variance. The component of interest is specified through its f_nu response vector
        response_a. The component to explicitly project out is specified through response_b.
        Derived from Eq 18 of arXiv:1006.5599
        """
        kmaps = self._prepare_maps(kmaps)
        response_a = self.responses[name1]
        response_b = self.responses[name2]
        brb = map_comb(response_b,response_b,self.cov,self.cinv)
        arb = map_comb(response_a,response_b,self.cov,self.cinv)
        arM = map_term(kmaps,response_a,self.cov,self.cinv)
        brM = map_term(kmaps,response_b,self.cov,self.cinv)
        ara = map_comb(response_a,response_a,self.cov,self.cinv)
        numer = brb * arM - arb*brM
        norm = (ara*brb-arb**2.)
        return numer/norm

def build_analytic_cov(ells,cmb_ps,fgdict,freqs,kbeams,noises,lmins=None,lmaxs=None,verbose=True):
    nmap = len(freqs)
    if cmb_ps.ndim==2: cshape = (nmap,nmap,1,1)
    elif cmb_ps.ndim==1: cshape = (nmap,nmap,1)
    else: raise ValueError
    Covmat = np.tile(cmb_ps,cshape)
    for i in range(nmap):
        for j in range(i,nmap):
            freq1 = freqs[i]
            freq2 = freqs[j]
            if verbose: print("Populating covariance for ",freq1,"x",freq2)
            for component in fgdict.keys():
                fgnoise = np.nan_to_num(fgdict[component](ells,freq1,freq2))
                fgnoise[np.abs(fgnoise)>1e90] = 0
                Covmat[i,j,...] = Covmat[i,j,...] + fgnoise
            Covmat[i,j,...] = Covmat[i,j,...] * kbeams[i] * kbeams[j]
            if i==j:
                Covmat[i,j,...] = Covmat[i,j,...] + noises[i]
                if lmins is not None: Covmat[i,j][ells<lmins[i]] = np.inf
                if lmaxs is not None: Covmat[i,j][ells>lmaxs[i]] = np.inf
            else: Covmat[j,i,...] = Covmat[i,j,...].copy()
    return Covmat


def standard_noise(response,cov=None,cinv=None):
    """
    Auto-noise <standard standard>
    """
    mcomb = map_comb(response,response,cov,cinv)
    return (1./mcomb)

def constrained_noise(response_a,response_b,cov=None,cinv=None,return_cross=True):
    """ Derived from Eq 18 of arXiv:1006.5599
    Auto-noise <constrained constrained>
    """
    brb = map_comb(response_b,response_b,cov,cinv)
    ara = map_comb(response_a,response_a,cov,cinv)
    arb = map_comb(response_a,response_b,cov,cinv)
    bra = map_comb(response_b,response_a,cov,cinv)
    numer = (brb)**2. * ara + (arb)**2.*brb - brb*arb*arb - arb*brb*bra
    denom = ara*brb-arb**2.
    d2 = (denom)**2.
    if return_cross:
        return (numer/d2), (brb*ara - arb*bra)/denom
    else:
        return (numer/d2)

def cross_noise(response_a,response_b,cov=None,cinv=None):
    """
    Cross-noise of <standard constrained>
    """
    snoise = standard_noise(response_a,cov,cinv)
    cnoise,cross = constrained_noise(response_a,response_b,cov,cinv,return_cross=True)
    return snoise*cross
    


def build_empirical_cov(ksplits,kcoadds,auto_for_cross_covariance=True,min_splits=None,fc=None):
    """
    Build an empirical covariance matrix using hybrid radial (signal) and cartesian
    (noise) binning.

    Args:

        ksplits: list of length narrays of (nsplits,Ny,Nx) fourier transforms of
        maps that have already been inverse noise weighted and tapered. This
        routine will not apply any corrections for the windowing.

        kcoadds: list of length narrays of (Ny,Nx) fourier transforms of
        coadd maps that have already been inverse noise weighted and tapered. This
        routine will not apply any corrections for the windowing.

        auto_for_cross_covariance: if True, cov(array1,array2) for array1!=array2 does
        not get separate signal and noise treatment, and is binned isotropically.
        TODO: allow for anisotropic binning of 90-150 correlation.

        min_splits: If not None, at least min_splits splits are required in an array
        for hybrid signal-noise treatment.

        fc: (optional) pre-initialized maps.FourierCalc object

    


    """
    narrays = len(ksplits)
    shape,wcs = ksplits[0].shape[-2:],ksplits[0].wcs
    if fc is None: fc = maps.FourierCalc(shape,wcs)
    Cov = maps.SymMat(narrays,shape)


    
    for aindex1 in range(narrays):
        for aindex2 in range(aindex1,narrays):
            if auto_for_cross_covariance and aindex1!=aindex2:
                scov = fc.f2power(kcoadds[aindex1],kcoadds[aindex2])
                ncov = None
            else:
                scov,ncov,autos = tutils.ncalc(ksplits,kcoadds,aindex1,aindex2,fc)
            if (min_splits is not None) and (aindex1==aindex2):
                nsplits = ksplits[aindex1].shape[0]
                if (nsplits<min_splits):
                    scov = autos
                    ncov = None
            if aindex1==aindex2:
                if ncov is None:
                    dncov = None
                elif noise_isotropic:
                    dncov = covtools.signal_average(ncov,bin_width=bin_width,kind=kind,lmin=lmins[aindex1])
                else:
                    # io.plot_img(maps.ftrans(ncov),aspect='auto')
                    dncov,_,_ = covtools.noise_average(ncov,dfact=dfact,
                                                       radial_fit=True if (tsim.nsplits[aindex1]==4 and atmosphere) else False,lmax=lmax,
                                                       wnoise_annulus=500,
                                                       lmin = 300,
                                                       bin_annulus=bin_width)
                    # io.plot_img(maps.ftrans(dncov),aspect='auto')
            else:
                dncov = None
            dscov = covtools.signal_average(scov,bin_width=bin_width,kind=kind,lmin=max(lmins[aindex1],lmins[aindex2])) # need to check this is not zero # ((a,inf),(inf,inf))  doesn't allow the first element to be used, so allow for cross-covariance from non informative
            #io.plot_img(maps.ftrans(dscov),aspect='auto')

            if dncov is None: dncov = np.zeros(dscov.shape)
            if aindex1==aindex2:
                dncov[modlmap<lmins[aindex1]] = np.inf
                dncov[modlmap>lmaxs[aindex1]] = np.inf
            Scov[aindex1,aindex2] = dscov[modlmap<lmax1].reshape(-1).copy()
            Ncov[aindex1,aindex2] = dncov[modlmap<lmax1].reshape(-1).copy()
            if aindex1!=aindex2: Scov[aindex2,aindex1] = Scov[aindex1,aindex2].copy()

    Cov = Scov + Ncov

    
