from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap,curvedsky
from enlib import bench
import numpy as np
import os,sys,shutil
from tilec import fg as tfg,ilc,kspace,utils as tutils
from soapack import interfaces as sints
from szar import foregrounds as fgs
import healpy as hp
from actsims.util import seed_tracker

"""

We will build a class specially suited for joint simulations
of ACT and Planck that cache in a way such that it is efficient
for a pipeline that runs on the same patch of sky for many sims.


We require the following inputs:
1. lensed alms of the CMB
2. a beam model for each ACT and Planck array
3. a model for the tSZ compton y as (nells,) PS
4. an (narray,narray,nells) PS covmat for the empirically fit residuals in each array 
where residual = power - noise - CMB model - tSZ model

Sim generation has three stages
Class initialization:
- load PS files
- load residual PS based on specified arrays
Index initialization:
- load lensed alm
- generate alm from compton y PS
- generate alm from residual PS
Compute map
- convolve with beam
- apply pixel window

"""


class JointSim(object):
    def __init__(self,qids,fg_res_version=None,ellmax=8101,bandpassed=True):
        self.qids = qids
        self.alms = {}
        ells = np.arange(ellmax)
        self.ellmax = ellmax
        self.cyy = fgs.power_y(ells)[None,None]
        self.cyy[0,0][ells<2] = 0

        if fg_res_version is not None:
            fpath = sints.dconfig['actsims']['fg_res_path'] + "/" + fg_res_version + "/"
            narrays = len(qids)
            self.cfgres = np.zeros((narrays,narrays,ellmax))
            for i in range(narrays):
                for j in range(i,narrays):
                    qid1 = qids[i]
                    qid2 = qids[j]
                    clfilename = "%sfgcov_%s_%s.txt" % (fpath,qid1,qid2)
                    clfilename_alt = "%sfgcov_%s_%s.txt" % (fpath,qid2,qid1)
                    try:
                        ls,cls = np.loadtxt(clfilename,unpack=True)
                    except:
                        ls,cls = np.loadtxt(clfilename_alt,unpack=True)
                    assert np.all(np.isclose(ls,ells))
                    self.cfgres[i,j] = cls.copy() 
                    if i!=j: self.cfgres[j,i] = cls.copy() 
            
        else:
            self.cfgres = None


        # get tSZ response, and also zero out parts of map that are not observed
        aspecs = tutils.ASpecs().get_specs
        bps = []
        for qid in qids:
            dm = sints.models[sints.arrays(qid,'data_model')]()
            if dm.name=='act_mr3':
                season,array1,array2 = sints.arrays(qid,'season'),sints.arrays(qid,'array'),sints.arrays(qid,'freq')
                array = '_'.join([array1,array2])
            elif dm.name=='planck_hybrid':
                season,patch,array = None,None,sints.arrays(qid,'freq')
            lmin,lmax,hybrid,radial,friend,cfreq,fgroup,wrfit = aspecs(qid)
            if bandpassed:
                bps.append("data/"+dm.get_bandpass_file_name(array))
            else:
                bps.append(cfreq)


        self.tsz_fnu = tfg.get_mix_bandpassed(bps, 'tSZ') if bandpassed else tfg.get_mix(bps, 'tSZ')




    def update_signal_index(self,sim_idx,set_idx=0,cmb_type='LensedUnabberatedCMB'):
        signal_path = sints.dconfig['actsims']['signal_path']
        cmb_file   = os.path.join(signal_path, 'fullsky%s_alm_set%02d_%05d.fits' %(cmb_type, set_idx, sim_idx))
        self.alms['cmb'] = hp.fitsfunc.read_alm(cmb_file, hdu = (1,2,3))
        comptony_seed = seed_tracker.get_fg_seed(set_idx, sim_idx, 'comptony')
        fgres_seed = seed_tracker.get_fg_seed(set_idx, sim_idx, 'srcfree')
        #self.alms['comptony'] = curvedsky.rand_alm_healpy(self.cyy, seed = comptony_seed)
        self.alms['comptony'] = curvedsky.rand_alm(self.cyy, lmax=self.ellmax, seed = comptony_seed) #!!!!
        if self.cfgres is not None: 
            #self.alms['fgres'] = curvedsky.rand_alm_healpy(self.cfgres, seed = fgres_seed)
            self.alms['fgres'] = curvedsky.rand_alm(self.cfgres, lmax=self.ellmax, seed = fgres_seed)


        # 1. convert to maximum ellmax
        lmax = max([hp.Alm.getlmax(self.alms[comp].shape[1]) for comp in self.alms.keys()])
        for comp in self.alms.keys():
            if hp.Alm.getlmax(self.alms[comp].shape[1])!=lmax: self.alms[comp] = maps.change_alm_lmax(self.alms[comp], lmax)
        self.lmax = lmax


    def compute_map(self,oshape,owcs,qid,pixwin_taper_deg=0.3,pixwin_pad_deg=0.3,
                    include_cmb=True,include_tsz=True,include_fgres=True,sht_beam=True):

        """
        1. get total alm
        2. apply beam, and pixel window if Planck
        3. ISHT
        4. if ACT, apply a small taper and apply pixel window in Fourier space
        """


        # pad to a slightly larger geometry
        tot_pad_deg = pixwin_taper_deg + pixwin_pad_deg
        res = maps.resolution(oshape,owcs)
        pix = np.deg2rad(tot_pad_deg)/res
        omap = enmap.pad(enmap.zeros((3,)+oshape,owcs),pix)
        ishape,iwcs = omap.shape[-2:],omap.wcs

        # get data model
        dm = sints.models[sints.arrays(qid,'data_model')](region_shape=ishape,region_wcs=iwcs,calibrated=True)

        # 1. get total alm
        array_index = self.qids.index(qid)
        tot_alm = int(include_cmb)*self.alms['cmb']
        tot_alm[0] = tot_alm[0] + int(include_tsz)*self.tsz_fnu[array_index]*self.alms['comptony'] 
        if self.cfgres is not None: tot_alm[0] = tot_alm[0] + int(include_fgres)*self.alms['fgres'][array_index]
        assert tot_alm.ndim==2
        assert tot_alm.shape[0]==3
        ells = np.arange(self.lmax+1)
        
        # 2. get beam, and pixel window for Planck
        if sht_beam:
            beam = tutils.get_kbeam(qid,ells,sanitize=False,planck_pixwin=False)    # NEVER SANITIZE THE BEAM IN A SIMULATION!!!
            for i in range(3): tot_alm[i] = hp.almxfl(tot_alm[i],beam)
            if dm.name=='planck_hybrid':
                pixwint,pixwinp = hp.pixwin(nside=tutils.get_nside(qid),lmax=self.lmax,pol=True)
                tot_alm[0] = hp.almxfl(tot_alm[0],pixwint)
                tot_alm[1] = hp.almxfl(tot_alm[1],pixwinp)
                tot_alm[2] = hp.almxfl(tot_alm[2],pixwinp)
        
        # 3. ISHT
        omap = curvedsky.alm2map(np.complex128(tot_alm),omap,spin=[0,2])
        assert omap.ndim==3
        assert omap.shape[0]==3


        if not(sht_beam):
            taper,_ = maps.get_taper_deg(ishape,iwcs,taper_width_degrees=pixwin_taper_deg,pad_width_degrees=pixwin_pad_deg)
            modlmap = omap.modlmap()
            beam = tutils.get_kbeam(qid,modlmap,sanitize=False,planck_pixwin=True)
            kmap = enmap.fft(omap*taper,normalize='phys')
            kmap = kmap * beam


        # 4. if ACT, apply a small taper and apply pixel window in Fourier space
        if dm.name=='act_mr3':
            if sht_beam: taper,_ = maps.get_taper_deg(ishape,iwcs,taper_width_degrees=pixwin_taper_deg,pad_width_degrees=pixwin_pad_deg)
            pwin = tutils.get_pixwin(ishape[-2:])
            if sht_beam: 
                omap = maps.filter_map(omap*taper,pwin)
            else:
                kmap = kmap * pwin

        if not(sht_beam): omap = enmap.ifft(kmap,normalize='phys').real

        return enmap.extract(omap,(3,)+oshape[-2:],owcs)


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
                       sim_splits=None,skip_inpainting=False,theory_signal="none",unsanitized_beam=False,save_all=False,plot_inpaint=False):


    save_scratch = not(memory_intensive)
    savedir = tutils.get_save_path(version,region)
    if save_scratch: 
        scratch = tutils.get_scratch_path(version,region)
        covscratch = scratch
    else:
        covscratch = None
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
        rfit_wnoise_widths = []
        save_names = [] # to make sure nothing is overwritten
        friends = {} # what arrays are each correlated with?
        names = arrays.split(',')
        print("Calculating FFTs for " , arrays)
        fbeam = lambda qname,x: tutils.get_kbeam(qname,x,sanitize=not(unsanitized_beam),planck_pixwin=True)
        for i,qid in enumerate(arrays.split(',')):
            dm = sints.models[sints.arrays(qid,'data_model')](region=mask,calibrated=not(uncalibrated))
            lmin,lmax,hybrid,radial,friend,cfreq,fgroup,wrfit = aspecs(qid)
            assert isinstance(radial,bool)
            do_radial_fit.append(radial)
            friends[qid] = friend
            hybrids.append(hybrid)
            freqs.append(fgroup)
            rfit_wnoise_widths.append(wrfit)
            kdiff,kcoadd,win = kspace.process(dm,region,qid,mask,
                                              skip_splits=False,
                                              splits_fname=sim_splits[i] if sim_splits is not None else None,
                                              inpaint=not(skip_inpainting),fn_beam = lambda x: fbeam(qid,x),
                                              plot_inpaint_path = savedir if plot_inpaint else None)
            print("Processed ",qid)
            if save_scratch: 
                kcoadd_name = savedir + "kcoadd_%s.npy" % qid
                kdiff_name = scratch + "kdiff_%s.npy" % qid
                win_name = scratch + "win_%s.npy" % qid
                assert win_name not in save_names
                assert kcoadd_name not in save_names
                assert kdiff_name not in save_names
                np.save(win_name,win)
                np.save(kcoadd_name,kcoadd)
                np.save(kdiff_name,kdiff)
                wins.append(win_name)
                kcoadds.append(kcoadd_name)
                kdiffs.append(kdiff_name)
                save_names.append(win_name)
                save_names.append(kcoadd_name)
                save_names.append(kdiff_name)
                print("Saved ",qid)
            else:
                wins.append(win.copy())
                kcoadds.append(kcoadd.copy())
                kdiffs.append(kdiff.copy())
            lmins.append(lmin)
            lmaxs.append(lmax)
    if sim_splits is not None: del sim_splits
    print("Done with ffts.")

    # print("Exiting because I just want to see inpainted stuff.")
    # sys.exit()

    # Decide what pairs to do hybrid smoothing for
    anisotropic_pairs = get_aniso_pairs(arrays.split(','),hybrids,friends)
    print("Anisotropic pairs: ",anisotropic_pairs)

    enmap.write_map(savedir+"tilec_mask.fits",mask)
    save_fn = lambda x,a1,a2: np.save(savedir+"tilec_hybrid_covariance_%s_%s.npy" % (names[a1],names[a2]),enmap.enmap(x,wcs))


    print("Building covariance...")
    with bench.show("build cov"):
        ilc.build_cov(names,kdiffs,kcoadds,fbeam,mask,lmins,lmaxs,freqs,anisotropic_pairs,
                      delta_ell,
                      do_radial_fit,save_fn,
                      signal_bin_width=signal_bin_width,
                      signal_interp_order=signal_interp_order,
                      rfit_lmaxes=lmaxs,
                      rfit_wnoise_widths=rfit_wnoise_widths,
                      rfit_lmin=rfit_lmin,
                      rfit_bin_width=None,
                      verbose=True,
                      debug_plots_loc=False,
                      separate_masks=False,theory_signal=theory_signal,scratch_dir=covscratch)


    if not(save_all): shutil.rmtree(scratch)



def build_and_save_ilc(arrays,region,version,cov_version,beam_version,
                       solutions,beams,chunk_size,
                       effective_freq,overwrite,maxval,unsanitized_beam=False,do_weights=False):

    print("Chunk size is ", chunk_size*64./8./1024./1024./1024., " GB.")
    def warn(): print("WARNING: no bandpass file found. Assuming array ",dm.c['id']," has no response to CMB, tSZ and CIB.")
    aspecs = tutils.ASpecs().get_specs
    bandpasses = not(effective_freq)
    savedir = tutils.get_save_path(version,region)
    covdir = tutils.get_save_path(cov_version,region)
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
        lmin,lmax,hybrid,radial,friend,cfreq,fgroup,wrfit = aspecs(qid)
        lmins.append(lmin)
        lmaxs.append(lmax)
        names.append(qid)
        if dm.name=='act_mr3':
            season,array1,array2 = sints.arrays(qid,'season'),sints.arrays(qid,'array'),sints.arrays(qid,'freq')
            array = '_'.join([array1,array2])
        elif dm.name=='planck_hybrid':
            season,patch,array = None,None,sints.arrays(qid,'freq')
        kcoadd_name = covdir + "kcoadd_%s.npy" % qid
        kmask = maps.mask_kspace(shape,wcs,lmin=lmin,lmax=lmax)
        kcoadd = enmap.enmap(np.load(kcoadd_name),wcs)
        dtype = kcoadd.dtype
        kcoadds.append(kcoadd.copy()*kmask)
        kbeam = tutils.get_kbeam(qid,modlmap,sanitize=not(unsanitized_beam),version=beam_version,planck_pixwin=True)
        kbeams.append(kbeam.copy())
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
            icov = enmap.enmap(np.load(covdir+"tilec_hybrid_covariance_%s_%s.npy" % (names[aindex1],names[aindex2])),wcs)
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
        if do_weights and len(comps)<=2:
            for qid in arrays:
                data[solution]['weight_%s' % qid] = enmap.zeros((Ny*Nx),wcs)
            

    for chunknum,(hilc,selchunk) in enumerate(ilcgen):
        print("ILC on chunk ", chunknum+1, " / ",int(modlmap.size/chunk_size)+1," ...")
        for solution in solutions:
            comps = data[solution]['comps']
            if len(comps)==1: # GENERALIZE
                data[solution]['noise'][selchunk] = hilc.standard_noise(comps[0])
                if do_weights: weight = hilc.standard_weight(comps[0])
                data[solution]['kmap'][selchunk] = hilc.standard_map(kcoadds[...,selchunk],comps[0])
            elif len(comps)==2:
                data[solution]['noise'][selchunk] = hilc.constrained_noise(comps[0],comps[1])
                data[solution]['cnoise'][selchunk] = hilc.cross_noise(comps[0],comps[1])
                ret = hilc.constrained_map(kcoadds[...,selchunk],comps[0],comps[1],return_weight=do_weights)
                if do_weights:
                    data[solution]['kmap'][selchunk],weight = ret
                else:
                    data[solution]['kmap'][selchunk] = ret

            elif len(comps)>2:
                data[solution]['kmap'][selchunk] = np.nan_to_num(hilc.multi_constrained_map(kcoadds[...,selchunk],comps[0],*comps[1:]))

            if len(comps)<=2 and do_weights:
                for qind,qid in enumerate(arrays):
                    data[solution]['weight_%s' % qid][selchunk] = weight[qind]


    del ilcgen,cov

    # Reshape into maps
    name_map = {'CMB':'cmb','tSZ':'comptony','CIB':'cib'}
    beams = beams.split(',')
    for solution,beam in zip(solutions,beams):
        comps = "tilec_single_tile_"+region+"_"
        comps = comps + name_map[data[solution]['comps'][0]]+"_"
        if len(data[solution]['comps'])>1: comps = comps + "deprojects_"+ '_'.join([name_map[x] for x in data[solution]['comps'][1:]]) + "_"
        comps = comps + version

        if do_weights and len(data[solution]['comps'])<=2:
            for qind,qid in enumerate(arrays):
                enmap.write_map("%s/%s_%s_weight.fits" % (savedir,comps,qid), enmap.enmap(data[solution]['weight_%s' % qid].reshape((Ny,Nx)),wcs))
            


        try:
            noise = enmap.enmap(data[solution]['noise'].reshape((Ny,Nx)),wcs)
            enmap.write_map("%s/%s_noise.fits" % (savedir,comps),noise)
        except: pass
        try:
            cnoise = enmap.enmap(data[solution]['cnoise'].reshape((Ny,Nx)),wcs)
            enmap.write_map("%s/%s_cross_noise.fits" % (savedir,comps),cnoise)
        except: pass

        ells = np.arange(0,modlmap.max(),1)
        try:
            fbeam = float(beam)
            kbeam = maps.gauss_beam(modlmap,fbeam)
            lbeam = maps.gauss_beam(ells,fbeam)
        except:
            qid = beam
            bfunc = lambda x: tutils.get_kbeam(qid,x,version=beam_version,sanitize=not(unsanitized_beam),planck_pixwin=False)
            kbeam = bfunc(modlmap)
            lbeam = bfunc(ells)

        kmap = enmap.enmap(data[solution]['kmap'].reshape((Ny,Nx)),wcs)
        smap = enmap.ifft(kbeam*kmap,normalize='phys').real
        enmap.write_map("%s/%s.fits" % (savedir,comps),smap)
        io.save_cols("%s/%s_beam.txt" % (savedir,comps),(ells,lbeam),header="ell beam")


    enmap.write_map(savedir+"/tilec_mask.fits",mask)
