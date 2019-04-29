import numpy as np
from pixell import utils,enmap
from tilec import covtools
from orphics import maps,stats,io

try: basestring
except NameError: basestring = str

"""
This module implements harmonic ILC.

TODO: add MCILC noise
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

def chunked_ilc(ells,kbeams,covfunc,chunk_size,responses=None,invert=True):
    """
    Provides a generator that can loop over chunks of fourier space
    and returns a HILC object for each.
    WARNING: This chunking mixes large and small scale modes, so it
    should only be used if you are sure your covariance matrix is
    well-behaved at all scales. FIXME: chunk in a sorted way.

    Args:
        kmaps: fourier transforms of tapered coadds
        of shape (narrays,Ny,Nx)
        covfunc: function(sel) that retuns a symmetric covariance matrix for that sel chunk
        chunk_size: number of fourier pixels in each chunk
        
    """
    nells = ells.size
    narrays = kbeams.shape[0]
    ls = ells.reshape(-1)
    kbeams = kbeams.reshape((narrays,nells))
    chunk_indices = range(0, nells, chunk_size)
    num_chunks = len(chunk_indices)
    for i in chunk_indices:
        selchunk = np.s_[i:i+chunk_size]
        hilc = HILC(ls[selchunk],kbeams[:,selchunk],covfunc(selchunk),responses=responses,invert=invert)
        yield hilc,selchunk

    

class HILC(object):
    """
    Harmonic ILC.
    We avoid beam deconvolution, instead modeling the beam in the response.
    Since all maps are beam convolved, we do not need lmaxes.
    """
    def __init__(self,ells,kbeams,cov=None,responses=None,invert=True):
        """
        Args:
            ells: (nells,) or (Ny,Nx) specifying mode number mapping for each pixel
            kbeams: (nmap,nells) or (nmap,Ny,Nx) fourier space beam factor. nmap
            determines number of frequencies/arrays used.
            cov: (nmap,nmap,nells) or (nmap,nmap,Ny,Nx) covariance matrix of 
            beam-convolved maps.
            responses: dictionary mapping component name to (nmap,) floats specifying
            the frequency/array response to that component for a beam-deconvolved map.
        """
        self.tol=1.e-4 #tolerance, please move elsewhere if desired
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
        self.kbeams = kbeams
        self.cov = np.moveaxis(cov,(0,1),(-2,-1))
        if np.any(np.isnan(self.cov)): 
            for i in range(self.cov.shape[-1]):
                for j in range(i,self.cov.shape[-1]):
                    print(ells[np.isnan(self.cov[...,i,j])])
            raise ValueError
        if invert:
            self.cinv = np.linalg.inv(self.cov) #utils.eigpow(np.nan_to_num(self.cov),-1,alim=0,rlim=0)
            self.cinv[self.ells<2] = 0
        else: self.cinv = None
        if np.any(np.isnan(self.cinv)): 
            print(ells.shape)
            print(self.cinv.shape)
            for i in range(self.cinv.shape[-1]):
                for j in range(i,self.cinv.shape[-1]):
                    print(ells[np.isnan(self.cinv[...,i,j])])
            print(self.cov[ells[np.isnan(self.cinv[...,i,j])].astype(np.int),...])
            raise ValueError
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
        snoise = self.standard_noise(name)
        snoise[np.isinf(np.abs(snoise))] = 0 # ells outside lmin and lmax are hopefully where the noise is inf
        out = weighted * snoise
        if np.any(np.isnan(out)): raise ValueError
        return out

    def constrained_map(self,kmaps,name1,name2):
        """Constrained ILC -- Make a constrained internal linear combination (ILC) of given fourier space maps at different frequencies
        and an inverse covariance matrix for its variance. The component of interest is specified through its f_nu response vector
        response_a. The component to explicitly project out is specified through response_b.
        Derived from Eq 18 of arXiv:1006.5599
        """
        kmaps = self._prepare_maps(kmaps)
        response_a = self.responses[name1]
        response_b = self.responses[name2]
        if np.any(np.isnan(response_a)): raise ValueError
        if np.any(np.isnan(kmaps)): raise ValueError
        if np.any(np.isnan(response_b)): raise ValueError
        if np.any(np.isnan(self.cinv)): raise ValueError
        brb = map_comb(response_b,response_b,self.cov,self.cinv)
        arb = map_comb(response_a,response_b,self.cov,self.cinv)
        arM = map_term(kmaps,response_a,self.cov,self.cinv)
        brM = map_term(kmaps,response_b,self.cov,self.cinv)
        ara = map_comb(response_a,response_a,self.cov,self.cinv)
        numer = brb * arM - arb*brM
        norm = 1./(ara*brb-arb**2.) 
        if np.any(np.isnan(numer)): raise ValueError
        norm[np.isinf(np.abs(norm))] = 0 # ells outside lmin and lmax are hopefully where the noise is inf
        out = numer*norm
        if np.any(np.isnan(out)): raise ValueError
        return out

    def multi_constrained_map(self,kmaps,name1,names=[]):
        """Multiply Constrained ILC -- Make a multiply constrained internal 
        linear combination (ILC) of given fourier space maps at different 
        frequencies
        and an inverse covariance matrix for its variance. The component of interest is specified through its f_nu response vector.  The
        components to explicitly project out are specified through a (arbitrarily-long, but not more than N_channels-1) list of responses."""
        kmaps = self._prepare_maps(kmaps)
        # compute the mixing tensor A_{p i \alpha}: this is the alpha^th component's SED evaluated for the i^th bandpass in Fourier pixel p
        N_comps = 1+len(names) #total number of components that are being explicitly modeled (one is preserved component)
        assert(N_comps < self.nmap) #ensure sufficient number of degrees of freedom
        A_mix = np.zeros((self.ells.size,self.nmap,N_comps))
        A_mix[:,:,0] = self.responses[name1] #component to be preserved -- always make this first column of mixing tensor
        for i,name in enumerate(names):
            assert(name != name1) #don't deproject the preserved component
            A_mix[:,:,i+1] = self.responses[name]
        # construct tensor Q_{p \alpha \beta} = (R^-1)_{p i j} A_{p i \alpha} A_{p j \beta}
        if self.cinv is not None:
            Qab = np.einsum('...ka,...kb->...ab',np.einsum('...ij,...ja->...ia',self.cinv,A_mix),A_mix)
        else:
            raise NotImplementedError
        # compute weights
        temp = np.zeros((self.ells.size,N_comps))
        if (N_comps == 1): # treat the no-deprojection case separately, since QSa is empty in this case
            temp[0] = 1.0
        else:
            for a in range(N_comps):
                QSa = np.delete(np.delete(Qab, a, -2), 0, -1) #remove the a^th row and zero^th column
                temp[:,a] = (-1.0)**float(a) * np.linalg.det(QSa)
        if self.cinv is not None:
            nweights = np.einsum('...ij,...i->...j',self.cinv,np.einsum('...a,...ia->...i',temp,A_mix))
        else:
            raise NotImplementedError
        weights = np.nan_to_num(1.0 / np.linalg.det(Qab)[:,None]) * nweights #FIXME: nan to num
        # verify responses
        diffs = np.absolute( np.sum(weights*A_mix[:,:,0],axis=-1) - 1. )
        # assert(np.all(diffs <= self.tol)) #preserved component FIXME: debug nans from det
        if (N_comps > 1):
            for i in range(1,N_comps):
                diffs = np.absolute( np.sum(weights*A_mix[:,:,i],axis=-1) )
                # assert(np.all(diffs <= self.tol)) #deprojected components FIXME: debug nans from det
        # apply weights to the data maps
        # N.B. total power of final ILC map in Fourier pixel p is: weights_{p i} Cov_{p i j} weights_{p j}
        return np.einsum('...i,...i->...',weights,kmaps)


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
                # if lmins is not None: Covmat[i,j][ells<lmins[i]] = 1e90
                # if lmaxs is not None: Covmat[i,j][ells>lmaxs[i]] = 1e90
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


def save_debug_plots(scov,dscov,ncov,dncov,tcov,modlmap,aindex1,aindex2,save_loc=None):
    if save_loc is None: save_loc = "."
    io.plot_img(maps.ftrans(scov),"%s/debug_s2d_%d_%d.png" % (save_loc,aindex1,aindex2),aspect='auto')
    io.plot_img(maps.ftrans(dscov),"%s/debug_ds2d_%d_%d.png" % (save_loc,aindex1,aindex2),aspect='auto')
    if ncov is not None:
        io.plot_img(maps.ftrans(ncov),"%s/debug_n2d_%d_%d.png" % (save_loc,aindex1,aindex2),aspect='auto')
        io.plot_img(maps.ftrans(dncov),"%s/debug_dn2d_%d_%d.png" % (save_loc,aindex1,aindex2),aspect='auto')
    bin_edges = np.arange(100,8000,100)
    binner = stats.bin2D(modlmap,bin_edges)
    cents = binner.centers
    #pl = io.Plotter(yscale='log',xlabel='$\\ell$',ylabel='$D_{\\ell}$',scalefn=lambda x:x**2./np.pi)
    pl = io.Plotter(xlabel='$\\ell$',ylabel='$D_{\\ell}$',scalefn=lambda x:x**2./np.pi)
    padd = lambda p,x,ls,col: p.add(cents,binner.bin(x)[1],ls=ls,color=col)
    padd(pl,scov,"-","C0")
    padd(pl,dscov,"--","C0")
    pl.done("%s/debug_s1d_%d_%d.png" % (save_loc,aindex1,aindex2))
    if ncov is not None: 
        pl = io.Plotter(yscale='log',xlabel='$\\ell$',ylabel='$D_{\\ell}$',scalefn=lambda x:x**2./np.pi)
        padd(pl,ncov,"-","C1")
        padd(pl,dncov,"--","C1")
        pl.done("%s/debug_n1d_%d_%d.png" % (save_loc,aindex1,aindex2))
    io.plot_img(maps.ftrans(tcov),"%s/debug_fcov2d_%d_%d.png" % (save_loc,aindex1,aindex2),aspect='auto')


def build_empirical_cov(ksplits,kcoadds,wins,mask,lmins,lmaxs,
                        anisotropic_pairs,do_radial_fit,save_fn,
                        signal_bin_width=None,
                        signal_interp_order=0,
                        dfact=(16,16),
                        rfit_lmaxes=None,
                        rfit_wnoise_width=250,
                        rfit_lmin=300,
                        rfit_bin_width=None,
                        verbose=True,
                        debug_plots_loc=None):
    """
    TODO: Add docs for wins and mask, and names and save_fn

    Build an empirical covariance matrix using hybrid radial (signal) and cartesian
    (noise) binning.

    Args:

        ksplits: list of length narrays of (nsplits,Ny,Nx) fourier transforms of
        maps that have already been inverse noise weighted and tapered. This
        routine will not apply any corrections for the windowing. Alternatively,
        list of strings containing filenames to versions of these on disk.

        kcoadds: list of length narrays of (Ny,Nx) fourier transforms of
        coadd maps that have already been inverse noise weighted and tapered. This
        routine will not apply any corrections for the windowing. Alternatively,
        list of strings containing filenames to versions of these on disk.


        anisotropic_pairs: list of 2-tuples specifying which elements of the covariance
        matrix will be treated under hybrid radial(signal)/cartesian(noise) mode. If
        (i,j) is in the list, (j,i) will be added if it already doesn't exist, since
        the covariance has to be symmetric. For these pairs, an atmospheric 1/f noise 
        will be fitted out before downsampling the noise power.
    

    For each pair of arrays, one of the following modes is chosen:
    1) fully radial
       - this is appropriate for any i,j for an isotropic experiment like Planck
         or for i!=j combinations of independent experiments like ACT/Planck
         or for i!=j combinations of ACT arrays that have independent instrument 
         noise (no correlated atmosphere), e.g. pa2,pa3
       - it is calculated simply by radially binning the total spectrum of
         the coadds ps(i,j). The split info is not used.
    
    2) hybrid radial/cartesian
       - this is appropriate for diagonals i=j of a ground-based experiment like
         ACT, or for i!=j combinations of ACT arrays that have correlated atmosphere
         e.g. pa3-150/pa3-90
       - this routine assumes that such off diagonal pairs of arrays have the same
         number of splits, and calculates the noise power from splits, and the signal from
         the total spectrum minus the noise power.

    """
    # Setup
    narrays = len(ksplits)
    on_disk = False
    try:
        shape,wcs = ksplits[0].shape[-2:],ksplits[0].wcs
    except:
        assert isinstance(ksplits[0],basestring), "List contents are neither enmaps nor filenames."
        shape,wcs = enmap.read_map_geometry(ksplits[0])
        shape = shape[-2:]
        on_disk = True
    def _load_map(kitem): return kitem if not(on_disk) else enmap.read_map(kitem)
    minell = maps.minimum_ell(shape,wcs)
    modlmap = enmap.modlmap(shape,wcs)

    # Defaults
    if rfit_lmaxes is None:
        px_arcmin = np.rad2deg(maps.resolution(shape,wcs))*60.
        rfit_lmaxes = [8000*0.5/px_arcmin]*narrays
    if rfit_bin_width is None: rfit_bin_width = minell*4.
    if signal_bin_width is None: signal_bin_width = minell*8.

    maxval = -np.inf
    # Loop over unique covmat elements
    for aindex1 in range(narrays):
        for aindex2 in range(aindex1,narrays):
            if verbose: print("Calculating covariance for array ", aindex1, " x ",aindex2, " ...")

            hybrid = ((aindex1,aindex2) in anisotropic_pairs) or ((aindex2,aindex1) in anisotropic_pairs)
            ks1 = _load_map(ksplits[aindex1])
            ks2 = _load_map(ksplits[aindex2])
            kc1 = _load_map(kcoadds[aindex1])
            kc2 = _load_map(kcoadds[aindex2])
            if kc1.ndim>2:
                assert kc1.shape[0]==1
                kc1 = kc1[0]
            if kc2.ndim>2:
                assert kc2.shape[0]==1
                kc2 = kc2[0]
            if hybrid:
                from actsims import noise as simnoise
                ncov = simnoise.noise_power(ks1,_load_map(wins[aindex1])*mask,
                                                 kmaps2=ks2,weights2=_load_map(wins[aindex2])*mask,
                                                 coadd_estimator=True)
                scov = np.real(kc1*kc2.conj())/np.mean(mask**2.) - ncov

            else:
                scov = np.real(kc1*kc2.conj())/np.mean(mask**2.)
                ncov = None

            dscov = covtools.signal_average(scov,bin_width=signal_bin_width,kind=signal_interp_order,lmin=max(lmins[aindex1],lmins[aindex2])) # ((a,inf),(inf,inf))  doesn't allow the first element to be used, so allow for cross-covariance from non informative
            if ncov is not None:
                dncov,_,_ = covtools.noise_average(ncov,dfact=dfact,
                                                   radial_fit=do_radial_fit[aindex1],lmax=max(rfit_lmaxes[aindex1],rfit_lmaxes[aindex2]),
                                                   wnoise_annulus=rfit_wnoise_width,
                                                   lmin = rfit_lmin,
                                                   bin_annulus=rfit_bin_width,fill_lmax=max(lmaxs[aindex1],lmaxs[aindex2]))
            else:
                dncov = np.zeros(dscov.shape)

            tcov = dscov + dncov
            assert np.all(np.isfinite(tcov))
            # print(modlmap[np.isinf(tcov)])
            # if aindex1==aindex2:
            #     tcov[modlmap<=lmins[aindex1]] = 1e90
            #     tcov[modlmap>=lmaxs[aindex1]] = 1e90

            maxcov = tcov.max()
            if maxcov>maxval: maxval = maxcov
            if np.any(np.isnan(tcov)): raise ValueError
            # save PS
            save_fn(tcov,aindex1,aindex2)
            if debug_plots_loc: save_debug_plots(scov,dscov,ncov,dncov,tcov,modlmap,aindex1,aindex2,save_loc=debug_plots_loc)
    return maxval
