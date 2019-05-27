from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
from enlib import bench
import numpy as np
import os,sys
from tilec import fg as tfg,ilc,kspace
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
                       signal_bin_width,signal_interp_order,dfact,
                       rfit_wnoise_width,rfit_lmin,
                       overwrite,memory_intensive,uncalibrated,
                       sim_splits=None):

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
    gconfig = io.config_from_yaml("input/data.yml")
    mask = sints.get_act_mr3_crosslinked_mask(region,
                                              version=mask_version,
                                              kind='binary_apod')
    shape,wcs = mask.shape,mask.wcs

    with bench.show("ffts"):
        kcoadds = []
        ksplits = []
        wins = []
        lmins = []
        lmaxs = []
        hybrids = []
        do_radial_fit = []
        save_names = [] # to make sure nothing is overwritten
        friends = {} # what arrays are each correlated with?
        names = []
        for i,array in enumerate(arrays.split(',')):
            ainfo = gconfig[array]
            dm = sints.models[ainfo['data_model']](region=mask,calibrated=not(uncalibrated))
            name = ainfo['id']
            names.append(name)
            rfit = ainfo['radial_fit']
            assert isinstance(rfit,bool)
            do_radial_fit.append(rfit)
            try: friends[name] = ainfo['correlated']
            except: friends[name] = None
            hybrids.append(ainfo['hybrid_average'])
            ksplit,kcoadd,win = kspace.process(dm,region,name,mask,
                                               ncomp=1,skip_splits=False,
                                               splits=sim_splits[i] if sim_splits is not None else None)
            if save_scratch: 
                kcoadd_name = savedir + "kcoadd_%s.hdf" % array
                ksplit_name = scratch + "ksplit_%s.hdf" % array
                win_name = scratch + "win_%s.hdf" % array
                assert win_name not in save_names
                assert kcoadd_name not in save_names
                assert ksplit_name not in save_names
                enmap.write_map(win_name,win)
                enmap.write_map(kcoadd_name,kcoadd)
                enmap.write_map(ksplit_name,ksplit)
                wins.append(win_name)
                kcoadds.append(kcoadd_name)
                ksplits.append(ksplit_name)
                save_names.append(win_name)
                save_names.append(kcoadd_name)
                save_names.append(ksplit_name)
            else:
                wins.append(win.copy())
                kcoadds.append(kcoadd.copy())
                ksplits.append(ksplit.copy())
            lmins.append(ainfo['lmin'])
            lmaxs.append(ainfo['lmax'])

    # Decide what pairs to do hybrid smoothing for
    anisotropic_pairs  = get_aniso_pairs(names,hybrids,friends)
    print("Anisotropic pairs: ",anisotropic_pairs)

    enmap.write_map(savedir+"tilec_mask.fits",mask)
    save_fn = lambda x,a1,a2: enmap.write_map(savedir+"tilec_hybrid_covariance_%s_%s.hdf" % (names[a1],names[a2]),enmap.enmap(x,wcs))

    maxval = ilc.build_empirical_cov(ksplits,kcoadds,wins,mask,lmins,lmaxs,
                            anisotropic_pairs,do_radial_fit,save_fn,
                            signal_bin_width=signal_bin_width,
                            signal_interp_order=signal_interp_order,
                            dfact=(dfact,dfact),
                            rfit_lmaxes=None,
                            rfit_wnoise_width=rfit_wnoise_width,
                            rfit_lmin=rfit_lmin,
                            rfit_bin_width=None,
                            verbose=True,
                            debug_plots_loc=savedir)

    np.savetxt(savedir + "maximum_value_of_covariances.txt",np.array([[maxval]]))



def build_and_save_ilc(arrays,region,version,cov_version,beam_version,
                       solutions,beams,chunk_size,
                       effective_freq,overwrite):

    print("Chunk size is ", chunk_size*64./8./1024./1024./1024., " GB.")
    def warn(): print("WARNING: no bandpass file found. Assuming array ",dm.c['id']," has no response to CMB, tSZ and CIB.")

    bandpasses = not(effective_freq)
    gconfig = io.config_from_yaml("input/data.yml")
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
    for i,array in enumerate(arrays):
        ainfo = gconfig[array]
        dm = sints.models[ainfo['data_model']](region=mask)
        array_id = ainfo['id']
        names.append(array_id)
        if dm.name=='act_mr3':
            season,array1,array2 = array_id.split('_')
            narray = array1 + "_" + array2
            patch = region
        elif dm.name=='planck_hybrid':
            season,patch,narray = None,None,array_id
        kcoadd_name = covdir + "kcoadd_%s.hdf" % array
        ainfo = gconfig[array]
        lmax = ainfo['lmax']
        lmin = ainfo['lmin']
        kmask = maps.mask_kspace(shape,wcs,lmin=lmin,lmax=lmax)
        kcoadd = enmap.read_map(kcoadd_name)
        dtype = kcoadd.dtype
        kcoadds.append(kcoadd.copy()*kmask)
        kbeams.append(dm.get_beam(ells=modlmap,season=season,patch=patch,array=narray,version=beam_version))
        if bandpasses:
            try: bps.append("data/"+ainfo['bandpass_file'] )
            except:
                warn()
                bps.append(None)
        else:
            try: bps.append(ainfo['bandpass_approx_freq'])
            except:
                warn()
                bps.append(None)

    kcoadds = enmap.enmap(np.stack(kcoadds),wcs)

    # Read Covmat
    maxval = np.loadtxt(covdir + "maximum_value_of_covariances.txt")
    assert maxval.size==1
    maxval = maxval.reshape(-1)[0]
    cov = maps.SymMat(narrays,shape[-2:])
    for aindex1 in range(narrays):
        for aindex2 in range(aindex1,narrays):
            icov = enmap.read_map(covdir+"tilec_hybrid_covariance_%s_%s.hdf" % (names[aindex1],names[aindex2]))
            cov[aindex1,aindex2] = icov

    cov.data = enmap.enmap(cov.data,wcs,copy=False)
    covfunc = lambda sel: cov.to_array(sel,flatten=True)

    assert cov.data.shape[0]==((narrays*(narrays+1))/2) # FIXME: generalize
    if np.any(np.isnan(cov.data)): raise ValueError 

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
        #comps = '_'.join(data[solution]['comps'])
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
            array = beam
            ainfo = gconfig[array]
            array_id = ainfo['id']
            dm = sints.models[ainfo['data_model']](region=mask)
            if dm.name=='act_mr3':
                season,array1,array2 = array_id.split('_')
                narray = array1 + "_" + array2
                patch = region
            elif dm.name=='planck_hybrid':
                season,patch,narray = None,None,array_id
            bfunc = lambda x: dm.get_beam(x,season=season,patch=patch,array=narray,version=beam_version)
            kbeam = bfunc(modlmap)
            lbeam = bfunc(ells)

        smap = enmap.ifft(kbeam*enmap.enmap(data[solution]['kmap'].reshape((Ny,Nx)),wcs),normalize='phys').real
        enmap.write_map("%s/%s.fits" % (savedir,comps),smap)
        io.save_cols("%s/%s_beam.txt" % (savedir,comps),(ells,lbeam),header="ell beam")


    enmap.write_map(savedir+"/tilec_mask.fits",mask)
