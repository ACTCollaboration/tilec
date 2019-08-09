import numpy as np
from pixell import utils,enmap
from tilec import covtools,fg as tfg,utils as tutils
from orphics import maps,stats,io,cosmology
from actsims import noise as simnoise
from szar import foregrounds as szfg
import os,sys
from enlib import bench
from scipy.optimize import curve_fit

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

def weight_term(response,cinv):
    """response^T . Cinv """
    return np.einsum('...i,...ij->...j',response,cinv)


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

    def standard_weight(self,name="cmb"):
        # Get normalized response^T cinv
        weighted = weight_term(self.responses[name],self.cinv).swapaxes(0,1)
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
    with np.errstate(divide='ignore'): ret = 1./mcomb
    ret[~np.isfinite(ret)] = 0
    return ret

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
    io.plot_img(maps.ftrans(scov),"%sdebug_s2d_%d_%d.png" % (save_loc,aindex1,aindex2),aspect='auto')
    io.plot_img(maps.ftrans(dscov),"%sdebug_ds2d_%d_%d.png" % (save_loc,aindex1,aindex2),aspect='auto')
    if ncov is not None:
        io.plot_img(maps.ftrans(ncov),"%sdebug_n2d_%d_%d.png" % (save_loc,aindex1,aindex2),aspect='auto')
        io.plot_img(maps.ftrans(dncov),"%sdebug_dn2d_%d_%d.png" % (save_loc,aindex1,aindex2),aspect='auto')
    bin_edges = np.arange(100,8000,100)
    binner = stats.bin2D(modlmap,bin_edges)
    cents = binner.centers
    #pl = io.Plotter(yscale='log',xlabel='$\\ell$',ylabel='$D_{\\ell}$',scalefn=lambda x:x**2./np.pi)
    pl = io.Plotter(xlabel='$\\ell$',ylabel='$D_{\\ell}$',scalefn=lambda x:x**2./np.pi)
    padd = lambda p,x,ls,col: p.add(cents,binner.bin(x)[1],ls=ls,color=col)
    padd(pl,scov,"-","C0")
    padd(pl,dscov,"--","C0")
    pl.done("%sdebug_s1d_%d_%d.png" % (save_loc,aindex1,aindex2))
    if ncov is not None: 
        pl = io.Plotter(yscale='log',xlabel='$\\ell$',ylabel='$D_{\\ell}$',scalefn=lambda x:x**2./np.pi)
        padd(pl,ncov,"-","C1")
        padd(pl,dncov,"--","C1")
        pl.done("%sdebug_n1d_%d_%d.png" % (save_loc,aindex1,aindex2))
    io.plot_img(maps.ftrans(tcov),"%sdebug_fcov2d_%d_%d.png" % (save_loc,aindex1,aindex2),aspect='auto',lim=[-5,1])



def _is_correlated(a1,a2,aids,carrays):
    c1s = carrays[a1] # list of ids of arrays correlated with a1
    c2s = carrays[a2] # list of ids of arrays correlated with a2
    if aids[a2] in c1s:
        assert aids[a1] in c2s
        return True
    else:
        assert aids[a1] not in c2s
        return False


class CTheory(object):
    def __init__(self,ells,cfile="input/cosmo2017_10K_acc3"):

        theory = cosmology.loadTheorySpectraFromCAMB(cfile,
                                                     unlensedEqualsLensed=False,
                                                     useTotal=False,
                                                     TCMB = 2.7255e6,
                                                     lpad=9000,get_dimensionless=False)
        self.ksz = szfg.power_ksz_reion(ells) + szfg.power_ksz_late(ells)
        self.cltt = theory.lCl('TT',ells)
        self.yy = szfg.power_y(ells)
        self.ells = ells
        self.dlscale = 2.*np.pi/self.ells/(self.ells+1.)
        self.dlscale[~np.isfinite(self.dlscale)] = 0


    def get_theory_cls(self,f1,f2,a_cmb=1,a_gal=0.8,exp_gal=-0.7): # a_gal is now 0.8 by default
        gf = lambda x: tfg.ItoDeltaT(x)
        clfg = szfg.power_tsz(self.ells,f1,f2,yy=self.yy) + szfg.power_cibp(self.ells,f1,f2) + szfg.power_cibc(self.ells,f1,f2) + \
               szfg.power_radps(self.ells,f1,f2) + self.ksz + a_gal * (self.ells/500.)**(exp_gal) * self.dlscale * (f1*f2/150./150.)**(3.8) * (gf(f1)*gf(f2)/gf(150.)**2.)
        return (a_cmb * self.cltt) + clfg

        

def scratch_fname(scratch_dir,ftype,a1,a2):
    return scratch_dir + "/%s_%d_%d.npy" % (ftype,a1,a2)


def build_cov_hybrid_coadd(names,kdiffs,kcoadds,fbeam,mask,
                           lmins,lmaxs,freqs,anisotropic_pairs,
                           delta_ell,
                           do_radial_fit,save_fn,
                           signal_bin_width=None,
                           signal_interp_order=0,
                           rfit_lmaxes=None,
                           rfit_wnoise_widths=250,
                           rfit_lmin=300,
                           rfit_bin_width=None,
                           verbose=True,
                           debug_plots_loc=None,separate_masks=False,theory_signal="none",
                           maxval=None,scratch_dir=None,fit_physical=None):

    """

    A more sophisticated covariance model. In comparison to deprecated.build_cov_hybrid, it doesn't simply
    use a different S+N model for each array, but rather coadds the signal S part across arrays
    that have similar frequencies and reuses the result for those arrays.

    We construct an (narray,narray,Ny,Nx) covariance matrix.
    We group each array by their rough frequency into nfreq groups.

    We loop through each a1,a2 array combination, each of which corresponds to an f1,f2 group combination.
    If a1==a2 or if they are correlated arrays, then we calculate noise from splits and store it into n(a1,a2).
    We calculate signal = (total - noise) and cache it into s(f1,f2).
    If a1!=a2 and they are not correlated arrays, then we calculate the cross of their coadds and cache it into s(f1,f2) and 
    set n(a1,a2) to zero.

    freqs: list of integers corresponding to central frequency denoting frequency group
    correlated_arrays: list of lists of correlated arrays
    fit_physical: if not None, specifies the integer ell below which the signal covariance is fit to a theoretical model
    """

    # General crap and input validation
    if fit_physical<2: fit_physical = None
    narrays = len(kdiffs)
    assert len(kcoadds)==len(lmins)==len(lmaxs)==len(freqs)==narrays

    # Is stuff on disk or in memory? Decide based on what's in the kdiffs list
    on_disk = False
    try:
        tshape = kdiffs[0].shape[-2:]
    except:
        assert isinstance(kdiffs[0],basestring), "List contents are neither enmaps nor filenames."
        on_disk = True

    # Mask as a function of array index is either the single mask or the separate mask
    # for each array
    def get_mask(aind):
        if separate_masks: 
            return _load_map(mask[aind])
        else: 
            assert mask.ndim==2
            return mask

    # Get geometry from the mask
    mask = get_mask(0)
    shape,wcs = mask.shape[-2:],mask.wcs

    # General map loading for both on disk and in memory
    def _load_map(kitem): 
        if not(on_disk):
            return kitem
        else:
            if kitem[-5:]=='.fits': return enmap.read_map(kitem)
            elif kitem[-4:]=='.npy': return enmap.enmap(np.load(kitem),wcs)
            else: raise IOError

    # Some useful geometry stuff
    minell = maps.minimum_ell(shape,wcs)
    modlmap = enmap.modlmap(shape,wcs)

    # Defaults
    if rfit_lmaxes is None:
        px_arcmin = np.rad2deg(maps.resolution(shape,wcs))*60.
        rfit_lmaxes = [8000*0.5/px_arcmin]*narrays
    if rfit_bin_width is None: rfit_bin_width = minell*4.
    if signal_bin_width is None: signal_bin_width = minell*8.
    if rfit_wnoise_widths is None: rfit_wnoise_widths = [250] * narrays


    # Let's build the instrument noise model
    if scratch_dir is None: dncovs = {} # where to store the smoothed noise
    if scratch_dir is None: scovs = {} # where to store the unsmoothed signal
    n1ds = {} # store the smoothed binned noise estimate for coadding of signal

    gellmax = modlmap.max()
    ells = np.arange(0,gellmax,1)

    # A fiducial theory object
    ctheory = CTheory(ells)

    # Loop through all the array combinations
    for a1 in range(narrays):
        for a2 in range(a1,narrays):

            # Load coadds and calculate total power
            m1 = get_mask(a1)
            m2 = get_mask(a2)
            kc1 = _load_map(kcoadds[a1])
            kc2 = _load_map(kcoadds[a2]) if a2!=a1 else kc1
            ccov = np.real(kc1*kc2.conj())/np.mean(m1*m2)

            # If off-diagonals that are not correlated, only calculate coadd cross for signal
            if (a1 != a2) and (not ((a1,a2) in anisotropic_pairs or (a2,a1) in anisotropic_pairs)): 
                scov = ccov
                if scratch_dir is None:
                    dncovs[(a1,a2)] = 0.
                else:
                    np.save(scratch_fname(scratch_dir,"dncovs",a1,a2),ccov * 0.)
            else:
                # Calculate noise power
                kd1 = _load_map(kdiffs[a1])
                kd2 = _load_map(kdiffs[a2])
                nsplits = kd1.shape[0]
                nsplits2 = kd2.shape[0]
                assert nsplits==nsplits2
                assert nsplits in [2,4], "Only two or four splits supported."
                with bench.show("noise power"):
                    ncov = simnoise.noise_power(kd1,m1,
                                                kmaps2=kd2,weights2=m2,
                                                coadd_estimator=True)

                # Smoothed noise power
                drfit = do_radial_fit[a1]
                if a1==a2: 
                    drfit = False # !!! only doing radial fit for 90-150
                    dolog = True
                else:
                    dolog = False
                
                with bench.show("noise smoothing"):
                    dncov,_,nparams = covtools.noise_block_average(ncov,nsplits=nsplits,delta_ell=delta_ell,
                                                                   radial_fit=drfit,lmax=min(min(rfit_lmaxes[a1],rfit_lmaxes[a2]),modlmap.max()),
                                                                   wnoise_annulus=min(rfit_wnoise_widths[a1],rfit_wnoise_widths[a2]),
                                                                   lmin = rfit_lmin,
                                                                   bin_annulus=rfit_bin_width,fill_lmax=min(min(lmaxs[a1],lmaxs[a2]),modlmap.max()),
                                                                   log=dolog)
                if scratch_dir is None:
                    dncovs[(a1,a2)] = dncov.copy()
                else:
                    np.save(scratch_fname(scratch_dir,"dncovs",a1,a2),dncov)
                if a1==a2:
                    # 1d approx of noise power for weights
                    if nparams is not None:
                        wfit,lfit,afit = nparams
                    else:
                        lmax = min(min(rfit_lmaxes[a1],rfit_lmaxes[a2]),modlmap.max())
                        rfit_wnoise_width = rfit_wnoise_widths[a1]
                        wfit = np.sqrt(dncov[np.logical_and(modlmap>=(lmax-rfit_wnoise_width),modlmap<lmax)].mean())*180.*60./np.pi
                        assert np.isfinite(wfit)
                        if tutils.is_planck(names[a1]): # !!!!
                            lfit = 0
                            afit = 1
                        else:
                            lfit = 3000
                            afit = -4
                    n1d = covtools.rednoise(ells,wfit,lfit,afit)
                    n1d[ells<2] = 0
                    n1ds[a1] = n1d.copy()
                    if verbose: print("Populating noise for %d,%d  (wnoise estimate of %.2f)" % (a1,a2,wfit))

                # signal power from coadd and unsmoothed noise power
                scov = ccov - ncov

            if theory_signal=="none":
                pass
            else:
                if (theory_signal=="diagonal" and a1==a2) or (theory_signal=="offdiagonal" and a1!=a2) or (theory_signal=="all") :
                    f1 = freqs[a1]
                    f2 = freqs[a2]
                    scov =  enmap.enmap(maps.interp(ells,ctheory.get_theory_cls(f1,f2))(modlmap) *fbeam(names[a1],modlmap) * fbeam(names[a2],modlmap) ,wcs)
                    print("WARNING: using theory signal for %d,%d" % (a1,a2))
                else:
                    if theory_signal not in ['none','diagonal','offdiagonal','all']: raise ValueError
            if scratch_dir is None:
                scovs[(a1,a2)] = scov.copy()
            else:
                np.save(scratch_fname(scratch_dir,"scovs",a1,a2),scov)

                

    fscovs = {}
    fws = {} # sum of weights in 2D
    for a1 in range(narrays):
        for a2 in range(a1,narrays):

            # Initialize signal cov and weights if needed
            f1 = freqs[a1]
            f2 = freqs[a2]
            if verbose: print("Calculating weights for %d,%d and frequencies %d,%d " % (a1,a2,f1,f2))
            try:
                fscovs[(f1,f2)]
            except:
                fscovs[(f1,f2)] = 0.
            try:
                fws[(f1,f2)]
            except:
                fws[(f1,f2)] = 0.

            c11 = ctheory.get_theory_cls(f1,f1)
            c22 = ctheory.get_theory_cls(f2,f2)
            c12 = ctheory.get_theory_cls(f1,f2)
            with np.errstate(divide='ignore'): cl_11 = c11 + n1ds[a1]/fbeam(names[a1],ells)**2.
            with np.errstate(divide='ignore'): cl_22 = c22 + n1ds[a2]/fbeam(names[a2],ells)**2.
            cl_12 = c12
            # cl_11[~np.isfinite(cl_11)] = 0
            # cl_22[~np.isfinite(cl_22)] = 0
            # cl_12[~np.isfinite(cl_12)] = 0 # !!!
            w = 1./((cl_11 * cl_22)+cl_12**2) 
            weight = maps.interp(ells,w)(modlmap)


            weight[modlmap<max(lmins[a1],lmins[a2])] = 0
            weight[modlmap>min(min(lmaxs[a1],lmaxs[a2]),modlmap.max())] = 0
            fws[(f1,f2)] = fws[(f1,f2)] + weight
            if scratch_dir is None:
                lscov = scovs[(a1,a2)]
            else:
                lscov = enmap.enmap(np.load(scratch_fname(scratch_dir,"scovs",a1,a2)),wcs)
            with np.errstate(divide='ignore',invalid='ignore'): 
                scov = lscov * weight / fbeam(names[a1],modlmap) / fbeam(names[a2],modlmap)
            scov[~np.isfinite(scov)] = 0
            fscovs[(f1,f2)] = fscovs[(f1,f2)] + scov

    slmin = min(lmins)

    # if fit_physical is not None:
    #     fitmax = fit_physical
    #     fitmin = slmin
    #     dell = 2*minell
    #     fbin_edges = np.arange(fitmin,fitmax,dell)
    #     fbinner = stats.bin2D(modlmap,fbin_edges)
    #     fcents = fbinner.centers
    #     ftheory = CTheory(fcents)
    mtheory = CTheory(modlmap)

    for a1 in range(narrays):
        for a2 in range(a1,narrays):
            f1 = freqs[a1]
            f2 = freqs[a2]

            key1 = (f1,f2)
            numer = fscovs[key1]
            denom = fws[key1]
            if f1!=f2:
                key2 = (f2,f1)
                try:
                    numer = numer + fscovs[key2]
                    denom = denom + fws[key2]
                    print("Found symmetric key for %d,%d" % (f1,f2))
                except:
                    pass
            with np.errstate(invalid='ignore'): nscov = numer/denom


            nscov[~np.isfinite(nscov)] = 0
            smsig = covtools.signal_average(nscov * fbeam(names[a1],modlmap) * fbeam(names[a2],modlmap),bin_width=signal_bin_width,
                                           kind=signal_interp_order,
                                           lmin=slmin,
                                           dlspace=True)

            # smsig = covtools.signal_average(nscov,bin_width=signal_bin_width,
            #                                kind=signal_interp_order,
            #                                lmin=slmin,
            #                                dlspace=True)  * fbeam(names[a1],modlmap) * fbeam(names[a2],modlmap) # !!!!

            # enmap.write_map("/scratch/r/rbond/msyriac/dump/unsmoothed_debeamed_%s_%s.fits" % (names[a1],names[a2]) ,enmap.enmap(nscov,wcs)) # !!!!
            # enmap.write_map("/scratch/r/rbond/msyriac/dump/beamsq_%s_%s.fits" % (names[a1],names[a2]) ,enmap.enmap(fbeam(names[a1],modlmap) * fbeam(names[a2],modlmap),wcs)) # !!!!


            # dtheory = mtheory.get_theory_cls(f1,f2) # !!!
            # pcov = nscov/dtheory
            # pcov[~np.isfinite(pcov)] = 0
            # dsig = covtools.signal_average(pcov* fbeam(names[a1],modlmap) * fbeam(names[a2],modlmap),bin_width=signal_bin_width,
            #                                 kind=signal_interp_order,
            #                                 lmin=slmin,
            #                                 dlspace=True) # !!!
            # smsig = dsig   * dtheory # !!!!

            # bin_edges = np.arange(20,3000,20)
            # binner = stats.bin2D(modlmap,bin_edges)
            # cents,d1d = binner.bin(dsig)

            # pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='r')
            # pl.add(cents,d1d)
            # pl.hline(y=1)
            # pl.done(os.environ['WORK']+"/flatsig_%s_%s.png" % (names[a1],names[a2]))

            # cents,d1d1 = binner.bin(smsig)
            # cents,d1d2 = binner.bin(dtheory * fbeam(names[a1],modlmap) * fbeam(names[a2],modlmap) )
            # pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='D',scalefn=lambda x: x**2./2./np.pi)
            # pl.add(cents,d1d1)
            # pl.add(cents,d1d2,ls="--")
            # pl._ax.set_ylim(1,1e4)
            # pl.done(os.environ['WORK']+"/dflatsig_%s_%s.png" % (names[a1],names[a2]))

            #enmap.write_map("/scratch/r/rbond/msyriac/dump/smoothed_beamed_%s_%s.fits" % (names[a1],names[a2]) ,enmap.enmap(smsig,wcs)) # !!!!

            # if fit_physical is not None:

            #     # np.random.seed(None)
            #     # cfit = smsig.copy() * (1+np.random.normal(scale=0.0001))
            #     # smsig[modlmap<fit_physical] = cfit[modlmap<fit_physical] 

            #     #enmap.write_map("/scratch/r/rbond/msyriac/dump/unsmoothed_beamed_%s_%s.fits" % (names[a1],names[a2]) ,enmap.enmap(nscov * fbeam(names[a1],modlmap) * fbeam(names[a2],modlmap),wcs))
            #     #enmap.write_map("/scratch/r/rbond/msyriac/dump/smoothed_beamed_%s_%s.fits" % (names[a1],names[a2]) ,enmap.enmap(smsig,wcs))
            #     #pass

            #     # ffunc = lambda d,x,y,z: ftheory.get_theory_cls(f1,f2,a_cmb=x,a_gal=y,exp_gal=z)
            #     # res,_ = curve_fit(ffunc,fcents,fbinner.bin(nscov)[1],p0=[1,0.8,-0.7],bounds=([0.2,0.,-2],[1.8,10.,0]))                
            #     # fcmb,fgal,fexp = res
            #     # cfit = maps.interp(ells,ctheory.get_theory_cls(f1,f2,a_cmb=fcmb,a_gal=fgal,exp_gal=fexp))(modlmap)
            #     # print(names[a1],names[a2],fcmb,fgal,fexp)

            #     # if True: #a1!=a2: 
            #     #     #cfit = maps.interp(ells,ctheory.get_theory_cls(f1,f2))(modlmap)*10
            #     #     smsig[modlmap<fit_physical] = cfit[modlmap<fit_physical] * fbeam(names[a1],modlmap[modlmap<fit_physical]) * fbeam(names[a2],modlmap[modlmap<fit_physical])
            #     pass
                


            smsig[modlmap<2] = 0

            # Diagnostic plot
            if debug_plots_loc: io.power_crop(smsig,200,debug_plots_loc+"dscov_%d_%d.png" % (a1,a2))
            if scratch_dir is None:
                lncov = dncovs[(a1,a2)]
            else:
                lncov = enmap.enmap(np.load(scratch_fname(scratch_dir,"dncovs",a1,a2)),wcs)
            enmap.write_map("/scratch/r/rbond/msyriac/dump/smoothed_noise_%s_%s.fits" % (names[a1],names[a2]) ,enmap.enmap(lncov,wcs)) # !!!


            # !!!!
            # ns = [240,290,41]
            # smsig = maps.interp(ells,ctheory.get_theory_cls(f1,f2))(modlmap) * fbeam(names[a1],modlmap) * fbeam(names[a2],modlmap) # !!!!
            # smsig[~np.isfinite(smsig)] = 0
            # !!!!



            ocov = lncov + smsig



            # !!!!
            # ns = [240,290,41]
            # ocov = maps.interp(ells,ctheory.get_theory_cls(f1,f2))(modlmap) * fbeam(names[a1],modlmap) * fbeam(names[a2],modlmap) + int(a1==a2)*(ns[a1]*np.pi/180./60.)**2.  # !!!!
            # ocov[~np.isfinite(ocov)] = 0
            # !!!!

            
            assert np.all(np.isfinite(ocov))
            if a1==a2:
                sel = np.logical_and(modlmap>=lmins[a1],modlmap<=lmaxs[a1])
                try:
                    assert np.all(ocov[sel]>0)
                except AssertionError:
                    ms = modlmap[sel][ocov[sel]<=0]
                    print(ms,ms.min(),ms.max())
                    raise



            if (maxval is not None) and a1==a2: 
                ocov[modlmap<lmins[a1]] = maxval
                ocov[modlmap>lmaxs[a2]] = maxval

            # Save S + N
            save_fn(ocov,a1,a2)
            if verbose: print("Populated final smoothed powers for %d,%d" % (a1,a2))




build_cov = build_cov_hybrid_coadd
