from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
from enlib import bench
import numpy as np
import os,sys
from tilec import fg as tfg,ilc,kspace,utils as tutils
from soapack import interfaces as sints

"""
Functions that produce outputs to disk as part of the pipeline.
These have been put into a module so that sims can be injected
seamlessly.
"""

def get_aniso_pairs(aids,hybrids,friends):
    narrays = len(aids)
    # Decide what pairs to do hybrid smoothing for
    anisotropic_pairs = []
    for i in range(narrays):
        for j in range(i,narrays):
            name1 = aids[i]
            name2 = aids[j]
            if (i==j) and (hybrids[i] and hybrids[j]):
                anisotropic_pairs.append((i,j))
                continue
            if (friends[name1] is None) or (friends[name2] is None): continue
            if name2 in friends[name1]:
                assert name1 in friends[name2], "Correlated arrays spec is not consistent."
                anisotropic_pairs.append((i,j))
    return anisotropic_pairs

def build_and_save_cov(arrays,region,version,mask_version,
                       signal_bin_width,signal_interp_order,delta_ell,
                       rfit_wnoise_width,rfit_lmin,
                       overwrite,memory_intensive,uncalibrated,
                       sim_splits=None,skip_inpainting=False,theory_signal=False):

    save_scratch = not(memory_intensive)
    save_path = sints.dconfig['tilec']['save_path']
    scratch_path = sints.dconfig['tilec']['scratch_path']
    savedir = save_path + version + "/" + region +"/"
    if save_scratch: scratch = scratch_path + version + "/" + region +"/"
    if not(overwrite):
        assert not(os.path.exists(savedir)), \
       "This version already exists on disk. Please use a different version identifier."
    try: os.makedirs(savedir)
    except:
        if overwrite: pass
        else: raise
    if save_scratch:     
        try: os.makedirs(scratch)
        except: pass
    mask = sints.get_act_mr3_crosslinked_mask(region,
                                              version=mask_version,
                                              kind='binary_apod')
    shape,wcs = mask.shape,mask.wcs
    aspecs = tutils.ASpecs().get_specs

    with bench.show("ffts"):
        kcoadds = []
        kdiffs = []
        wins = []
        lmins = []
        lmaxs = []
        hybrids = []
        do_radial_fit = []
        freqs = []
        save_names = [] # to make sure nothing is overwritten
        friends = {} # what arrays are each correlated with?
        names = arrays.split(',')
        print(arrays)
        for i,qid in enumerate(arrays.split(',')):
            dm = sints.models[sints.arrays(qid,'data_model')](region=mask,calibrated=not(uncalibrated))
            lmin,lmax,hybrid,radial,friend,cfreq,fgroup = aspecs(qid)
            assert isinstance(radial,bool)
            do_radial_fit.append(radial)
            friends[qid] = friend
            hybrids.append(hybrid)
            freqs.append(fgroup)
            fbeam = lambda qname,x: tutils.get_kbeam(qname,x)
            kdiff,kcoadd,win = kspace.process(dm,region,qid,mask,
                                              skip_splits=False,
                                              splits=sim_splits[i] if sim_splits is not None else None,
                                              inpaint=not(skip_inpainting),fn_beam = lambda x: fbeam(qid,x))
            if save_scratch: 
                kcoadd_name = savedir + "kcoadd_%s.hdf" % qid
                kdiff_name = scratch + "kdiff_%s.hdf" % qid
                win_name = scratch + "win_%s.hdf" % qid
                assert win_name not in save_names
                assert kcoadd_name not in save_names
                assert kdiff_name not in save_names
                enmap.write_map(win_name,win)
                enmap.write_map(kcoadd_name,kcoadd)
                enmap.write_map(kdiff_name,kdiff)
                wins.append(win_name)
                kcoadds.append(kcoadd_name)
                kdiffs.append(kdiff_name)
                save_names.append(win_name)
                save_names.append(kcoadd_name)
                save_names.append(kdiff_name)
            else:
                wins.append(win.copy())
                kcoadds.append(kcoadd.copy())
                kdiffs.append(kdiff.copy())
            lmins.append(lmin)
            lmaxs.append(lmax)

    # Decide what pairs to do hybrid smoothing for
    anisotropic_pairs = get_aniso_pairs(arrays.split(','),hybrids,friends)
    print("Anisotropic pairs: ",anisotropic_pairs)

    enmap.write_map(savedir+"tilec_mask.fits",mask)
    save_fn = lambda x,a1,a2: enmap.write_map(savedir+"tilec_hybrid_covariance_%s_%s.hdf" % (names[a1],names[a2]),enmap.enmap(x,wcs))



    with bench.show("build cov"):
        ilc.build_cov(names,kdiffs,kcoadds,fbeam,mask,lmins,lmaxs,freqs,anisotropic_pairs,
                  delta_ell,
                  do_radial_fit,save_fn,
                  signal_bin_width=signal_bin_width,
                  signal_interp_order=signal_interp_order,
                  rfit_lmaxes=lmaxs,
                  rfit_wnoise_width=rfit_wnoise_width,
                  rfit_lmin=rfit_lmin,
                  rfit_bin_width=None,
                  verbose=True,
                  debug_plots_loc=False,
                  separate_masks=False,theory_signal=theory_signal)






def build_and_save_ilc(arrays,region,version,cov_version,beam_version,
                       solutions,beams,chunk_size,
                       effective_freq,overwrite,maxval):

    print("Chunk size is ", chunk_size*64./8./1024./1024./1024., " GB.")
    def warn(): print("WARNING: no bandpass file found. Assuming array ",dm.c['id']," has no response to CMB, tSZ and CIB.")
    aspecs = tutils.ASpecs().get_specs
    bandpasses = not(effective_freq)
    save_path = sints.dconfig['tilec']['save_path']
    savedir = save_path + version + "/" + region +"/"
    covdir = save_path + cov_version + "/" + region +"/"
    assert os.path.exists(covdir)
    if not(overwrite):
        assert not(os.path.exists(savedir)), \
       "This version already exists on disk. Please use a different version identifier."
    try: os.makedirs(savedir)
    except:
        if overwrite: pass
        else: raise


    mask = enmap.read_map(covdir+"tilec_mask.fits")
    shape,wcs = mask.shape,mask.wcs
    Ny,Nx = shape
    modlmap = enmap.modlmap(shape,wcs)

    arrays = arrays.split(',')
    narrays = len(arrays)
    kcoadds = []
    kbeams = []
    bps = []
    names = []
    lmins = []
    lmaxs = []
    for i,qid in enumerate(arrays):
        dm = sints.models[sints.arrays(qid,'data_model')](region=mask,calibrated=True)
        lmin,lmax,hybrid,radial,friend,cfreq,fgroup = aspecs(qid)
        lmins.append(lmin)
        lmaxs.append(lmax)
        names.append(qid)
        if dm.name=='act_mr3':
            season,array1,array2 = sints.arrays(qid,'season'),sints.arrays(qid,'array'),sints.arrays(qid,'freq')
            array = '_'.join([array1,array2])
        elif dm.name=='planck_hybrid':
            season,patch,array = None,None,sints.arrays(qid,'freq')
        kcoadd_name = covdir + "kcoadd_%s.hdf" % qid
        kmask = maps.mask_kspace(shape,wcs,lmin=lmin,lmax=lmax)
        kcoadd = enmap.read_map(kcoadd_name)
        dtype = kcoadd.dtype
        kcoadds.append(kcoadd.copy()*kmask)
        kbeams.append(dm.get_beam(ells=modlmap,season=season,patch=region,array=array,version=beam_version))
        if bandpasses:
            try: bps.append("data/"+dm.get_bandpass_file_name(array) )
            except:
                warn()
                bps.append(None)
        else:
            try: bps.append(cfreq)
            except:
                warn()
                bps.append(None)

    kcoadds = enmap.enmap(np.stack(kcoadds),wcs)



    # Read Covmat
    cov = maps.SymMat(narrays,shape[-2:])
    for aindex1 in range(narrays):
        for aindex2 in range(aindex1,narrays):
            icov = enmap.read_map(covdir+"tilec_hybrid_covariance_%s_%s.hdf" % (names[aindex1],names[aindex2]))
            if aindex1==aindex2: 
                icov[modlmap<lmins[aindex1]] = maxval
                icov[modlmap>lmaxs[aindex1]] = maxval
            cov[aindex1,aindex2] = icov
    cov.data = enmap.enmap(cov.data,wcs,copy=False)
    covfunc = lambda sel: cov.to_array(sel,flatten=True)

    assert cov.data.shape[0]==((narrays*(narrays+1))/2) # FIXME: generalize
    assert np.all(np.isfinite(cov.data))

    # Make responses
    responses = {}
    for comp in ['tSZ','CMB','CIB']:
        if bandpasses:
            responses[comp] = tfg.get_mix_bandpassed(bps, comp)
        else:
            responses[comp] = tfg.get_mix(bps, comp)

    ilcgen = ilc.chunked_ilc(modlmap,np.stack(kbeams),covfunc,chunk_size,responses=responses,invert=True)

    # Initialize containers
    solutions = solutions.split(',')
    data = {}
    kcoadds = kcoadds.reshape((narrays,Ny*Nx))
    for solution in solutions:
        data[solution] = {}
        comps = solution.split('-')
        data[solution]['comps'] = comps
        if len(comps)<=2: 
            data[solution]['noise'] = enmap.zeros((Ny*Nx),wcs)
        if len(comps)==2: 
            data[solution]['cnoise'] = enmap.zeros((Ny*Nx),wcs)
        data[solution]['kmap'] = enmap.zeros((Ny*Nx),wcs,dtype=dtype) # FIXME: reduce dtype?

    for chunknum,(hilc,selchunk) in enumerate(ilcgen):
        print("ILC on chunk ", chunknum+1, " / ",int(modlmap.size/chunk_size)+1," ...")
        for solution in solutions:
            comps = data[solution]['comps']
            if len(comps)==1: # GENERALIZE
                data[solution]['noise'][selchunk] = hilc.standard_noise(comps[0])
                data[solution]['kmap'][selchunk] = hilc.standard_map(kcoadds[...,selchunk],comps[0])
            elif len(comps)==2:
                data[solution]['noise'][selchunk] = hilc.constrained_noise(comps[0],comps[1])
                data[solution]['cnoise'][selchunk] = hilc.cross_noise(comps[0],comps[1])
                data[solution]['kmap'][selchunk] = hilc.constrained_map(kcoadds[...,selchunk],comps[0],comps[1])
            elif len(comps)>2:
                data[solution]['kmap'][selchunk] = np.nan_to_num(hilc.multi_constrained_map(kcoadds[...,selchunk],comps[0],*comps[1:]))

    del ilcgen,cov

    # Reshape into maps
    name_map = {'CMB':'cmb','tSZ':'comptony','CIB':'cib'}
    beams = beams.split(',')
    for solution,beam in zip(solutions,beams):
        comps = "tilec_single_tile_"+region+"_"
        comps = comps + name_map[data[solution]['comps'][0]]+"_"
        if len(data[solution]['comps'])>1: comps = comps + "deprojects_"+ '_'.join([name_map[x] for x in data[solution]['comps'][1:]]) + "_"
        comps = comps + version
        try:
            noise = enmap.enmap(data[solution]['noise'].reshape((Ny,Nx)),wcs)
            enmap.write_map("%s/%s_noise.fits" % (savedir,comps),noise)
        except: pass
        try:
            cnoise = enmap.enmap(data[solution]['cnoise'].reshape((Ny,Nx)),wcs)
            enmap.write_map("%s/%s_cross_noise.fits" % (savedir,comps),noise)
        except: pass

        ells = np.arange(0,modlmap.max(),1)
        try:
            fbeam = float(beam)
            kbeam = maps.gauss_beam(modlmap,fbeam)
            lbeam = maps.gauss_beam(ells,fbeam)
        except:
            qid = beam
            bfunc = lambda x: tutils.get_kbeam(qid,x,version=beam_version)
            kbeam = bfunc(modlmap)
            lbeam = bfunc(ells)

        kmap = enmap.enmap(data[solution]['kmap'].reshape((Ny,Nx)),wcs)
        smap = enmap.ifft(kbeam*kmap,normalize='phys').real
        enmap.write_map("%s/%s.fits" % (savedir,comps),smap)
        io.save_cols("%s/%s_beam.txt" % (savedir,comps),(ells,lbeam),header="ell beam")


    enmap.write_map(savedir+"/tilec_mask.fits",mask)
