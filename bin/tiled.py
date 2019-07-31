"""
Tiled logic

- Have a list of array geometries in memory
- Loop through each array
- Check if center of tile falls in array geometry
- If not, skip array
- Extract full 8 degree tile from ivar of array
- if any pixel in central 4 degree of tile is missing (ivar=0), skip array
- Extract full 8 degree tile from map of array


We still need to apodize-mask the sharp edges of arrays that enter outside the central region of the tile.


Check filter_div

"""
import os,sys
import numpy as np
from soapack import interfaces as sints
from pixell import enmap
from tilec import tiling,kspace,ilc,pipeline,fg as tfg
from orphics import mpi, io,maps
from enlib import bench
from tilec.utils import coadd,is_planck,apodize_zero,get_splits,get_splits_ivar,robust_ref,filter_div,get_kbeam,load_geometries,get_specs
comm = mpi.MPI.COMM_WORLD

maxval = 700000
chunk_size = 1000000
bandpasses = False
solutions = ['CMB','tSZ']
#solutions = ['CMB']#,'tSZ']
beams = ['2.0','2.0']
#beams = ['7.0','7.0']

# args
ivar_apod_pix = 120
#ivar_apod_pix = 60
#qids = ['p01','p02','p03','p04','p07','p08']
# qids = ['p01','p02','p03','p04','p05','p06','p07','p08']
#qids = ['p04','p07','p08']
#qids = ['p01','p02','p03','p04','p05','p06','p07','p08']
#qids = ['p01','p07']#,'p08']
#qids = ['s16_04','s16_03']
#qids = ['d56_04','d56_05','d56_06']#,'s16_01','s16_02','s16_03']
        # 'd56_02',
        # 'd56_03',
        # 'd56_04',
        # 'd56_05',
        # 'd56_06'] # list of quick IDs
#qids = ['p05','s16_01','s16_02','s16_03'] # list of quick IDs
# qids = ['p05','d5',
#         'd6',
#         'd56_01',
#         'd56_02',
#         'd56_03',
#         'd56_04',
#         'd56_05',
#         'd56_06'] # list of quick IDs
# qids = ['s16_01','s16_02','s16_03'] # list of quick IDs
#qids = ['p01','p02','p03','p04','p05','p06','p07','p08','s16_01','s16_02','s16_03']
qids = ['d5',
        'd6',
        'd56_01',
        'd56_02',
        'd56_03',
        'd56_04',
        'd56_05',
        'd56_06','s16_01','s16_02','s16_03','p01','p02','p03','p04','p05','p06','p07','p08'] # list of quick IDs
# qids = ['d5',
#         'd6',
#         'd56_01',
#         'd56_02',
#         'd56_03',
#         'd56_04',
#         'd56_05',
#         'd56_06','s16_01','s16_02','s16_03','p04','p05','p06','p07','p08'] # list of quick IDs
# qids = ['d5',
#         'd6',
#         'd56_01',
#         'd56_02',
#         'd56_03',
#         'd56_04',
#         'd56_05',
#         'd56_06','s16_01','s16_02','s16_03','p04','p05'] # list of quick IDs
parent_qid = 'd56_01' # qid of array whose geometry will be used for the full map




pdefaults = io.config_from_yaml("input/cov_defaults_tiled.yml")['cov']

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("--dtiles",     type=str,  default=None,help="A description.")
parser.add_argument("--theory",     type=str,  default="none",help="A description.")
parser.add_argument("--onlyd", action='store_true',help='A flag.')
parser.add_argument("--ivars", action='store_true',help='A flag.')
parser.add_argument("--signal-bin-width",     type=int,  default=pdefaults['signal_bin_width'],help="A description.")
parser.add_argument("--signal-interp-order",     type=int,  default=pdefaults['signal_interp_order'],help="A description.")
parser.add_argument("--delta-ell",     type=int,  default=pdefaults['delta_ell'],help="A description.")
parser.add_argument("--rfit-bin-width",     type=int,  default=pdefaults['rfit_bin_width'],help="A description.")
parser.add_argument("--rfit-wnoise-width",     type=int,  default=pdefaults['rfit_wnoise_width'],help="A description.")
parser.add_argument("--rfit-lmin",     type=int,  default=pdefaults['rfit_lmin'],help="A description.")

args = parser.parse_args()




geoms = load_geometries(qids)
pshape,pwcs = load_geometries([parent_qid])[parent_qid]
# mask = sints.get_act_mr3_crosslinked_mask("deep56")
# pshape,pwcs = mask.shape,mask.wcs



ta = tiling.TiledAnalysis(pshape,pwcs,comm=comm,width_deg=4.,pix_arcmin=0.5)
for solution in solutions:
    ta.initialize_output(name=solution)
down = lambda x,n=2: enmap.downgrade(x,n)


if args.dtiles is not None:
    dtiles = [int(x) for x in args.dtiles.split(',')]
else:
    dtiles = []



for i,extracter,inserter,eshape,ewcs in ta.tiles(from_file=True): # this is an MPI loop
    # What is the shape and wcs of the tile? is this needed?
    aids = [] ; kdiffs = [] ; ksplits = [] ; kcoadds = [] ; masks = []
    lmins = [] ; lmaxs = [] ; do_radial_fit = [] ; hybrids = [] ; friends = {}
    bps = [] ; kbeams = [] ; freqs = [] 
    modlmap = enmap.modlmap(eshape,ewcs)

    
    if args.onlyd:
        if i not in dtiles: continue

        
    for qid in qids:
        # Check if this array is useful
        ashape,awcs = geoms[qid]
        Ny,Nx = ashape[-2:]
        center = enmap.center(eshape,ewcs)
        acpixy,acpixx = enmap.sky2pix(ashape,awcs,center)
        # Following can be made more restrictive by being aware of tile shape
        if acpixy<=0 or acpixx<=0 or acpixy>=Ny or acpixx>=Nx: continue
        # Ok so the center of the tile is inside this array, but are there any missing pixels?
        eivars = get_splits_ivar(qid,extracter)
        # Only check for missing pixels if array is not a Planck array
        if eivars is None: continue
        if not(is_planck(qid)) and ("s16" not in qid) and np.any(ta.crop_main(eivars)<=0): 
            print("Skipping %s as it seems to have some zeros in the tile center..." % qid)
            continue
        aids.append(qid)
        apod = ta.apod * apodize_zero(np.sum(eivars,axis=0),ivar_apod_pix)
        esplits = get_splits(qid,extracter)

        if args.ivars:
            if i in dtiles:
                #io.hplot(esplits * apod,os.environ['WORK']+"/tiling/esplits_%s_%d" % (qid,i))
                #io.hplot(eivars * apod,os.environ['WORK']+"/tiling/eivars_%s_%d" % (qid,i))
                io.plot_img(esplits * apod,os.environ['WORK']+"/tiling/esplits_%s_%d" % (qid,i))
                div = eivars.copy()
                div[div<=0] = np.nan
                io.plot_img(div * apod,os.environ['WORK']+"/tiling/eivars_%s_%d" % (qid,i))
        
        kdiff,kcoadd = kspace.process_splits(esplits,eivars,apod,skip_splits=False,do_fft_splits=False)
        kdiffs.append(kdiff.copy())
        # ksplits.append(ksplit.copy())
        lmin,lmax,hybrid,radial,friend,cfreq,fgroup = get_specs(qid)
        freqs.append(fgroup)
        dtype = kcoadd.dtype
        kcoadds.append(kcoadd.copy())
        masks.append(apod.copy())
        lmins.append(lmin)
        lmaxs.append(lmax)
        hybrids.append(hybrid)
        do_radial_fit.append(radial)
        friends[qid] = friend
        bps.append(cfreq) # change to bandpass file
        kbeams.append(get_kbeam(qid,modlmap))
    fbeam = lambda qname,x: get_kbeam(qname,x)
        
    if len(aids)==0: continue # this tile is empty
    # Then build the covmat placeholder
    narrays = len(aids)
    cov = maps.SymMat(narrays,eshape[-2:])
    def save_fn(x,a1,a2): cov[a1,a2] = enmap.enmap(x,ewcs).copy()
    print(comm.rank, ": Tile %d has arrays " % i, aids)
    anisotropic_pairs = pipeline.get_aniso_pairs(aids,hybrids,friends)
    def stack(x): return enmap.enmap(np.stack(x),ewcs)


    kcoadds = stack(kcoadds)
    masks = stack(masks)


    with bench.show("cov"):
        ilc.build_cov(names=qids,kdiffs=kdiffs,kcoadds=kcoadds,fbeam=fbeam,
                      mask=masks,lmins=lmins,lmaxs=lmaxs,freqs=freqs,
                      anisotropic_pairs=anisotropic_pairs,
                      delta_ell=args.delta_ell,
                      do_radial_fit=do_radial_fit,save_fn=save_fn,
                      signal_bin_width=args.signal_bin_width,
                      signal_interp_order=args.signal_interp_order,
                      rfit_lmaxes=lmaxs,
                      rfit_wnoise_width=args.rfit_wnoise_width,
                      rfit_lmin=args.rfit_lmin,
                      rfit_bin_width=None,
                      verbose=True,
                      debug_plots_loc=os.environ['WORK'] + '/tiling/dplots_tile_%d_' % i if i in dtiles else False,
                      # debug_plots_loc=False,#os.environ['WORK'] + '/tiling/dplots_tile_%d_' % i if i in [18,19,44,69] else False,
                      separate_masks=True,maxval=maxval,theory_signal=args.theory)



    cov.data = enmap.enmap(cov.data,ewcs,copy=False)
    covfunc = lambda sel: cov.to_array(sel,flatten=True)
    assert cov.data.shape[0]==((narrays*(narrays+1))/2) # FIXME: generalize
    assert np.all(np.isfinite(cov.data))

    # bps, kbeams, 

    # Make responses
    responses = {}
    for comp in ['tSZ','CMB','CIB']:
        if bandpasses:
            responses[comp] = tfg.get_mix_bandpassed(bps, comp)
        else:
            responses[comp] = tfg.get_mix(bps, comp)
    ilcgen = ilc.chunked_ilc(modlmap,np.stack(kbeams),covfunc,chunk_size,responses=responses,invert=True)
    Ny,Nx = eshape[-2:]

    # Initialize containers
    data = {}

    for qind,qid in enumerate(aids):
        lmin = lmins[qind]
        lmax = lmaxs[qind]
        kmask = maps.mask_kspace(eshape,ewcs,lmin=lmin,lmax=lmax)
        kcoadds[qind] = kcoadds[qind] * kmask



    # !!!!
    # dy = 45
    # dx = 19
    # c1 = cov.to_array(flatten=False)[:,:,dy,dx]
    # c2 = cov.to_array(flatten=False)[:,:,dy-1,dx]
    # v1 = kcoadds[:,dy,dx]
    # v2 = kcoadds[:,dy-1,dx]
    # print(np.abs(v1))
    # print(np.abs(v2))
    # # np.save("cov_bad_pixel.npy",c1)
    # # np.save("cov_good_pixel.npy",c2)
    # # np.save("vec_bad_pixel.npy",v1)
    # # np.save("vec_good_pixel.npy",v2)
    # print(np.linalg.eig(c1)[0])
    # print(np.linalg.eig(c2)[0])
    # sys.exit()
    # !!!!



    kcoadds = kcoadds.reshape((narrays,Ny*Nx))
    for solution in solutions:
        data[solution] = {}
        comps = solution.split('-')
        data[solution]['comps'] = comps
        if len(comps)<=2: 
            data[solution]['noise'] = enmap.zeros((Ny*Nx),ewcs)
        if len(comps)==2: 
            data[solution]['cnoise'] = enmap.zeros((Ny*Nx),ewcs)
        data[solution]['kmap'] = enmap.zeros((Ny*Nx),ewcs,dtype=dtype) # FIXME: reduce dtype?

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
    for solution,beam in zip(solutions,beams):
        # comps = "tilec_single_tile_"+region+"_"
        # comps = comps + name_map[data[solution]['comps'][0]]+"_"
        # if len(data[solution]['comps'])>1: comps = comps + "deprojects_"+ '_'.join([name_map[x] for x in data[solution]['comps'][1:]]) + "_"
        # comps = comps + version
        # try:
        #     noise = enmap.enmap(data[solution]['noise'].reshape((Ny,Nx)),ewcs)
        #     enmap.write_map("%s/%s_noise.fits" % (savedir,comps),noise)
        # except: pass
        # try:
        #     cnoise = enmap.enmap(data[solution]['cnoise'].reshape((Ny,Nx)),ewcs)
        #     enmap.write_map("%s/%s_cross_noise.fits" % (savedir,comps),noise)
        # except: pass

        ells = np.arange(0,modlmap.max(),1)
        try:
            fbeam = float(beam)
            kbeam = maps.gauss_beam(modlmap,fbeam)
            lbeam = maps.gauss_beam(ells,fbeam)
        except:
            raise
            # array = beam
            # ainfo = gconfig[array]
            # array_id = ainfo['id']
            # dm = sints.models[ainfo['data_model']](region=mask)
            # if dm.name=='act_mr3':
            #     season,array1,array2 = array_id.split('_')
            #     narray = array1 + "_" + array2
            #     patch = region
            # elif dm.name=='planck_hybrid':
            #     season,patch,narray = None,None,array_id
            # bfunc = lambda x: dm.get_beam(x,season=season,patch=patch,array=narray,version=beam_version)
            # kbeam = bfunc(modlmap)
            # lbeam = bfunc(ells)

        ksol = data[solution]['kmap'].reshape((Ny,Nx))
        assert np.all(np.isfinite(ksol))
        ksol[modlmap<min(lmins)] = 0
        # ksol[modlmap<2200] = 0  #!!!
        # ksol[modlmap>2250] = 0  #!!!
        # print(ksol[0,4])
        # print(ksol[0,5])
        # sys.exit()
        # if solution=='CMB' and i in dtiles: 
        #     optile = np.real(ksol*ksol.conj())
        #     ptile = np.real(ksol*ksol.conj())
        #     ptile[modlmap>2250] = 0
        #     ptile[modlmap<2200] = 0
        #     px = enmap.argmax(ptile,unit='pix')
        #     print(px)
        #     print(ksol[px[0],px[1]])
        #     print(ksol[px[0]-11,px[1]])
        #     print(optile[px[0],px[1]])
        #     print(optile[px[0]-1,px[1]])
        #     print(modlmap[px[0],px[1]])
        #     print(modlmap[px[0]-1,px[1]])
        #     # pftile = ptile
        #     # pftile[modlmap>300] = 0
        #     # print(np.sort(pftile[pftile>0]))
        #     # print(modlmap[np.isclose(ptile,1.52256073e+02)])
        #     # io.plot_img(np.log10(np.fft.fftshift(ptile)),os.environ['WORK']+"/tiling/ptile_%d_smap" % i)
        #     # io.hplot(enmap.enmap(np.log10(np.fft.fftshift(ptile)),ewcs),os.environ['WORK']+"/tiling/phtile_%d_smap" % i)
        smap = enmap.ifft(kbeam*enmap.enmap(ksol,ewcs),normalize='phys').real
        if solution=='CMB': 
            io.hplot(smap,os.environ['WORK']+"/tiling/tile_%d_smap" % i)
            io.power_crop(np.real(ksol*ksol.conj()),100,os.environ['WORK']+"/tiling/ptile_%d.png" % i)
        # sys.exit()
        ta.update_output(solution,smap,inserter)
    #ta.update_output("processed",c*civar,inserter)
    #ta.update_output("processed_ivar",civar,inserter)
    #pmap = ilc.do_ilc
    #ta.update_output("processed",pmap,inserter)
print("Rank %d done" % comm.rank)
for solution in solutions:
    pmap = ta.get_final_output(solution)
    if comm.rank==0:
        io.hplot(pmap,os.environ['WORK']+"/tiling/map_%s" % solution)
        mask = sints.get_act_mr3_crosslinked_mask("deep56")
        io.hplot(enmap.extract(pmap,mask.shape,mask.wcs)*mask,os.environ['WORK']+"/tiling/mmap_%s" % solution)
    




