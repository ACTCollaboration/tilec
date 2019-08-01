

def build_empirical_cov(kdiffs,kcoadds,wins,mask,lmins,lmaxs,
                        anisotropic_pairs,do_radial_fit,save_fn,
                        signal_bin_width=None,
                        signal_interp_order=0,
                        delta_ell=400,
                        rfit_lmaxes=None,
                        rfit_wnoise_width=250,
                        rfit_lmin=300,
                        rfit_bin_width=None,
                        verbose=True,
                        debug_plots_loc=None,separate_masks=False):
    """
    TODO: Add docs for wins and mask, and names and save_fn

    Build an empirical covariance matrix using hybrid radial (signal) and cartesian
    (noise) binning.

    Args:

        kdiffs: list of length narrays of (nsplits,Ny,Nx) fourier transforms of
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
    narrays = len(kdiffs)
    on_disk = False
    try:
        tshape = kdiffs[0].shape[-2:]
    except:
        assert isinstance(kdiffs[0],basestring), "List contents are neither enmaps nor filenames."
        on_disk = True

    def get_mask(aind):
        if separate_masks: 
            return _load_map(mask[aind])
        else: 
            assert mask.ndim==2
            return mask

    mask = get_mask(0)
    shape,wcs = mask.shape[-2:],mask.wcs

    def _load_map(kitem): 
        if not(on_disk):
            return kitem
        else:
            if kitem[:-5]=='.fits': return enmap.read_map(kitem)
            elif kitem[:-4]=='.npy': return enmap.enmap(np.load(kitem),wcs)
            else: raise IOError
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


            # !!!!!
            # from orphics import cosmology
            # theory = cosmology.default_theory()
            # # #['p01','p02','p03','p04','p05','p06','p07','p08']
            # # freqs = [30,44,70,100,143,217,353,545]
            # # fwhms = [7*143/freq for freq in freqs]
            # # rmss = [195,226,199,77,33,47,153,1000]
            # # fwhm1 = fwhms[aindex1]
            # # fwhm2 = fwhms[aindex2]
            # rms = 0 if aindex1!=aindex2 else 60 #rmss[aindex1]
            # fwhm1 = fwhm2 = 1.5
            # tcov = theory.lCl('TT',modlmap)*maps.gauss_beam(modlmap,fwhm1)*maps.gauss_beam(modlmap,fwhm2) + rms # !!!!
            # save_fn(tcov,aindex1,aindex2)
            # continue
            # !!!

            if verbose: print("Calculating covariance for array ", aindex1, " x ",aindex2, " ...")

            hybrid = ((aindex1,aindex2) in anisotropic_pairs) or ((aindex2,aindex1) in anisotropic_pairs)
            kd1 = _load_map(kdiffs[aindex1])
            kd2 = _load_map(kdiffs[aindex2])
            nsplits1 = kd1.shape[0]
            nsplits2 = kd2.shape[0]
            assert (nsplits1==2 or nsplits1==4) and (nsplits2==2 or nsplits2==4)
            kc1 = _load_map(kcoadds[aindex1])
            kc2 = _load_map(kcoadds[aindex2])
            if kc1.ndim>2:
                assert kc1.shape[0]==1
                kc1 = kc1[0]
            if kc2.ndim>2:
                assert kc2.shape[0]==1
                kc2 = kc2[0]
            m1 = get_mask(aindex1)
            m2 = get_mask(aindex2)
            if hybrid:
                w1 = _load_map(wins[aindex1])
                w2 = _load_map(wins[aindex2])
                ncov = simnoise.noise_power(kd1,m1,
                                                 kmaps2=kd2,weights2=m2,
                                                 coadd_estimator=True)
                ccov = np.real(kc1*kc2.conj())/np.mean(m1*m2)
                scov = ccov - ncov


                from orphics import cosmology
                theory = cosmology.default_theory()
                # # #['p01','p02','p03','p04','p05','p06','p07','p08']
                # # freqs = [30,44,70,100,143,217,353,545]
                # # fwhms = [7*143/freq for freq in freqs]
                # # rmss = [195,226,199,77,33,47,153,1000]
                # # fwhm1 = fwhms[aindex1]
                # # fwhm2 = fwhms[aindex2]
                # rms = 0 if aindex1!=aindex2 else 60 #rmss[aindex1]
                fwhm1 = fwhm2 = 1.5
                scov = theory.lCl('TT',modlmap)*maps.gauss_beam(modlmap,fwhm1)*maps.gauss_beam(modlmap,fwhm2)
                # save_fn(tcov,aindex1,aindex2)
                

                # import os
                # # newcov = np.real(kd1[aindex1][0]*ksplits[aindex2][1].conj())/np.mean(w1*w2*m1*m2)
                # enmap.write_map(os.environ['WORK']+'/tiling/ncov.fits',ncov)
                # enmap.write_map(os.environ['WORK']+'/tiling/ccov.fits',ccov)
                # enmap.write_map(os.environ['WORK']+'/tiling/scov.fits',scov)
                # sys.exit()

            else:
                scov = np.real(kc1*kc2.conj())/np.mean(m1*m2)
                ncov = None

            dscov = covtools.signal_average(scov,bin_width=signal_bin_width,kind=signal_interp_order,lmin=max(lmins[aindex1],lmins[aindex2]),dlspace=True) # ((a,inf),(inf,inf))  doesn't allow the first element to be used, so allow for cross-covariance from non informative
            if ncov is not None:
                assert nsplits1==nsplits2
                nsplits = nsplits1
                dncov,_,_ = covtools.noise_block_average(ncov,nsplits=nsplits,delta_ell=delta_ell,
                                                         radial_fit=do_radial_fit[aindex1],lmax=max(rfit_lmaxes[aindex1],rfit_lmaxes[aindex2]),
                                                         wnoise_annulus=rfit_wnoise_width,
                                                         lmin = rfit_lmin,
                                                         bin_annulus=rfit_bin_width,fill_lmax=max(lmaxs[aindex1],lmaxs[aindex2]),
                                                         log=(aindex1==aindex2))
            else:
                dncov = np.zeros(dscov.shape)

            tcov = dscov + dncov






            assert np.all(np.isfinite(tcov))
            # print(modlmap[np.isinf(tcov)])
            if aindex1==aindex2:
                tcov[modlmap<=lmins[aindex1]] = 1e90
                tcov[modlmap>=lmaxs[aindex1]] = 1e90


            # try:
            #     print("scov : ",scov[4,959],scov[5,959])
            # except: pass
            # try:
            #     print("ncov : ",ncov[4,959],ncov[5,959])
            # except:
            #     pass
            # print("dscov : ",dscov[4,959],dscov[5,959])
            # print("dncov : ",dncov[4,959],dncov[5,959])


            maxcov = tcov.max()
            if maxcov>maxval: maxval = maxcov
            if np.any(np.isnan(tcov)): raise ValueError
            # save PS
            save_fn(tcov,aindex1,aindex2)
            if debug_plots_loc: save_debug_plots(scov,dscov,ncov,dncov,tcov,modlmap,aindex1,aindex2,save_loc=debug_plots_loc)
    return maxval


def build_cov_hybrid(names,kdiffs,kcoadds,fbeam,mask,lmins,lmaxs,freqs,anisotropic_pairs,delta_ell,
              do_radial_fit,save_fn,
              signal_bin_width=None,
              signal_interp_order=0,
              rfit_lmaxes=None,
              rfit_wnoise_widths=None,
              rfit_lmin=300,
              rfit_bin_width=None,
              verbose=True,
              debug_plots_loc=None,separate_masks=False,theory_signal=False):

    """

    A hybrid covariance model: it calculate noise spectra from difference maps and signal spectra
    from the coadd power minus noise estimate. It then smooths the noise spectra with block averaging
    and the signal spectra with annular binning.

    """

    narrays = len(kdiffs)
    assert len(kcoadds)==len(lmins)==len(lmaxs)==len(freqs)

    on_disk = False
    try:
        shape,wcs = kdiffs[0].shape[-2:],kdiffs[0].wcs
    except:
        assert isinstance(kdiffs[0],basestring), "List contents are neither enmaps nor filenames."
        shape,wcs = enmap.read_map_geometry(kdiffs[0])
        shape = shape[-2:]
        on_disk = True
    def _load_map(kitem): 
        if not(on_disk):
            return kitem
        else:
            if kitem[:-5]=='.fits': return enmap.read_map(kitem)
            elif kitem[:-4]=='.npy': return enmap.enmap(np.load(kitem),wcs)
            else: raise IOError
    minell = maps.minimum_ell(shape,wcs)
    modlmap = enmap.modlmap(shape,wcs)
    def get_mask(aind):
        if separate_masks: 
            return _load_map(mask[aind])
        else: 
            assert mask.ndim==2
            return mask

    # Defaults
    if rfit_lmaxes is None:
        px_arcmin = np.rad2deg(maps.resolution(shape,wcs))*60.
        rfit_lmaxes = [8000*0.5/px_arcmin]*narrays
    if rfit_bin_width is None: rfit_bin_width = minell*4.
    if signal_bin_width is None: signal_bin_width = minell*8.
    if rfit_wnoise_widths is None: rfit_wnoise_widths = [250] * narrays


    # Let's build the instrument noise model
    gellmax = max(lmaxs)
    ells = np.arange(0,gellmax,1)
    ctheory = CTheory(ells)
    for a1 in range(narrays):
        for a2 in range(a1,narrays):
            f1 = freqs[a1]
            f2 = freqs[a2]
            # Skip off-diagonals that are not correlated
            if (a1 != a2) and (not ((a1,a2) in anisotropic_pairs or (a2,a1) in anisotropic_pairs)): 
                continue
            kd1 = _load_map(kdiffs[a1])
            kd2 = _load_map(kdiffs[a2])
            nsplits = kd1.shape[0]
            nsplits2 = kd2.shape[0]
            assert nsplits==nsplits2
            assert nsplits in [2,4], "Only two or four splits supported."
            m1 = get_mask(a1)
            m2 = get_mask(a2)
            ncov = simnoise.noise_power(kd1,m1,
                                        kmaps2=kd2,weights2=m2,
                                        coadd_estimator=True)

            dncov,_,nparams = covtools.noise_block_average(ncov,nsplits=nsplits,delta_ell=delta_ell,
                                                                    radial_fit=do_radial_fit[a1],lmax=min(rfit_lmaxes[a1],rfit_lmaxes[a2]),
                                                                    wnoise_annulus=min(rfit_wnoise_widths[a1],rfit_wnoise_widths[a2]),
                                                                    lmin = rfit_lmin,
                                                                    bin_annulus=rfit_bin_width,fill_lmax=min(lmaxs[a1],lmaxs[a2]),
                                                                    log=(a1==a2))


            savg = lambda x: covtools.signal_average(x,bin_width=signal_bin_width,
                                                     kind=signal_interp_order,
                                                     lmin=max(lmins[a1],lmins[a2]),
                                                     dlspace=True)

            kc1 = _load_map(kcoadds[a1])
            kc2 = _load_map(kcoadds[a2]) if a2!=a1 else kc1
            ccov = np.real(kc1*kc2.conj())/np.mean(m1*m2)
            if (a1 != a2) and (not (((a1,a2) in anisotropic_pairs) or ((a2,a1) in anisotropic_pairs))): 
                scov = ccov
            else:
                scov = ccov - ncov
            smsig = savg(scov)


            if theory_signal and (a1==a2):
                smsig =  maps.interp(ells,ctheory.get_theory_cls(f1,f2)*fbeam(names[a1],ells) * fbeam(names[a2],ells))(modlmap) # !!!
                smsig[~np.isfinite(smsig)] = 0


            fcov = smsig + dncov
            save_fn(fcov,a1,a2)

            if verbose: print("Populating noise for %d,%d belonging to freqs %d,%d" % (a1,a2,f1,f2))
                

def build_cov_isotropic(names,kdiffs,kcoadds,fbeam,mask,lmins,lmaxs,freqs,anisotropic_pairs,delta_ell,
              do_radial_fit,save_fn,
              signal_bin_width=None,
              signal_interp_order=0,
              rfit_lmaxes=None,
              rfit_wnoise_width=250,
              rfit_lmin=300,
              rfit_bin_width=None,
              verbose=True,
              debug_plots_loc=None,separate_masks=False,theory_signal=False):

    """
    The simplest covariance model: it just bins all spectra in annuli.
    """

    narrays = len(kdiffs)
    assert len(kcoadds)==len(lmins)==len(lmaxs)==len(freqs)

    on_disk = False
    try:
        shape,wcs = kdiffs[0].shape[-2:],kdiffs[0].wcs
    except:
        assert isinstance(kdiffs[0],basestring), "List contents are neither enmaps nor filenames."
        shape,wcs = enmap.read_map_geometry(kdiffs[0])
        shape = shape[-2:]
        on_disk = True
    def _load_map(kitem): 
        if not(on_disk):
            return kitem
        else:
            if kitem[:-5]=='.fits': return enmap.read_map(kitem)
            elif kitem[:-4]=='.npy': return enmap.enmap(np.load(kitem),wcs)
            else: raise IOError
    minell = maps.minimum_ell(shape,wcs)
    modlmap = enmap.modlmap(shape,wcs)
    def get_mask(aind):
        if separate_masks: 
            return _load_map(mask[aind])
        else: 
            assert mask.ndim==2
            return mask

    # Defaults
    if rfit_lmaxes is None:
        px_arcmin = np.rad2deg(maps.resolution(shape,wcs))*60.
        rfit_lmaxes = [8000*0.5/px_arcmin]*narrays
    if rfit_bin_width is None: rfit_bin_width = minell*4.
    if signal_bin_width is None: signal_bin_width = minell*8.


    # Let's build the instrument noise model
    gellmax = max(lmaxs)
    ells = np.arange(0,gellmax,1)
    ctheory = CTheory(ells)
    for a1 in range(narrays):
        for a2 in range(a1,narrays):
            f1 = freqs[a1]
            f2 = freqs[a2]
            m1 = get_mask(a1)
            m2 = get_mask(a2)
            savg = lambda x: covtools.signal_average(x,bin_width=signal_bin_width,
                                                     kind=signal_interp_order,
                                                     lmin=max(lmins[a1],lmins[a2]),
                                                     dlspace=True)
            kc1 = _load_map(kcoadds[a1])
            kc2 = _load_map(kcoadds[a2]) if a2!=a1 else kc1
            ccov = np.real(kc1*kc2.conj())/np.mean(m1*m2)
            smsig = savg(ccov)
            fcov = smsig
            save_fn(fcov,a1,a2)
            if verbose: print("Populating noise for %d,%d belonging to freqs %d,%d" % (a1,a2,f1,f2))
                

