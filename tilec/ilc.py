import numpy as np

"""
This module implements harmonic ILC.
"""

def map_comb(response_a,response_b,cov):
    """Return a^T Cinv b"""
    # Cinv b = np.linalg.solve(cov,b)
    # Cov is in shape (...n,n)
    Cinvb = np.linalg.solve(cov,response_b)
    return np.nan_to_num(np.einsum('...l,...l->...',response_a,Cinvb))

def map_term(kmaps,cov,response):
    """response^T . Cinv . kmaps """
    Cinvk = np.linalg.solve(cov,kmaps)
    return np.einsum('...k,...k->...',response,Cinvk)

class HILC(object):
    """
    Harmonic ILC.
    We avoid beam deconvolution, instead modeling the beam in the response.
    Since all maps are beam convolved, we do not need lmaxes.
    """
    def __init__(self,ells,kbeams,cov=None,responses=None,chunks=1):
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
        self.responses = {}
        if "cmb" not in responses.keys(): responses['cmb'] = np.ones((1,nmap))
        for key in responses.keys():
            self.add_response(key,responses[key].reshape((1,nmap)))
                
    def add_response(self,name,response):
        self.responses[name] = response * self.kbeams

    def standard_noise(self,name):
        r = self.responses[name]
        mcomb = map_comb(r,r,self.cov)
        return (1./mcomb)

    def constrained_noise(self,name1,name2):
        """ Derived from Eq 18 of arXiv:1006.5599 """
        response_a = self.responses[name1]
        response_b = self.responses[name2]
        brb = map_comb(response_b,response_b,self.cov)
        ara = map_comb(response_a,response_a,self.cov)
        arb = map_comb(response_a,response_b,self.cov)
        bra = map_comb(response_b,response_a,self.cov)
        numer = (brb)**2. * ara + (arb)**2.*brb - brb*arb*arb - arb*brb*bra
        denom = (ara*brb-arb**2.)**2.
        return (numer/denom)

    def _prepare_maps(self,kmaps):
        assert kmaps.shape[0] == self.nmap
        if self._2d: kmaps = kmaps.reshape((self.nmap,self.ells.size))
        kmaps = kmaps.swapaxes(0,1)
        return kmaps
        
    def standard_map(self,kmaps,name="cmb"):
        # Get response^T cinv kmaps
        kmaps = self._prepare_maps(kmaps)
        weighted = map_term(kmaps,self.cov,self.responses[name])
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
        brb = map_comb(response_b,response_b,self.cov)
        arb = map_comb(response_a,response_b,self.cov)
        arM = map_term(kmaps,self.cov,response_a)
        brM = map_term(kmaps,self.cov,response_b)
        ara = map_comb(response_a,response_a,self.cov)
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
