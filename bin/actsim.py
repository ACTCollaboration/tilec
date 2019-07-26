from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap
import numpy as np
import os,sys
from actsims import noise as actnoise
from actsims.util import seed_tracker
from soapack import interfaces as sints
from enlib import bench
from tilec import pipeline,utils as tutils



pdefaults = io.config_from_yaml("input/cov_defaults.yml")['cov']

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("version", type=str,help='Version name.')
parser.add_argument("region", type=str,help='Region name.')
parser.add_argument("arrays", type=str,help='Comma separated list of array names. Array names map to a data specification in data.yml')
parser.add_argument("solutions", type=str,help='Comma separated list of solutions. Each solution is of the form x-y-... where x is solved for and the optionally provided y-,... are deprojected. The x can belong to any of CMB,tSZ and y,z,... can belong to any of CMB,tSZ,CIB.')
parser.add_argument("beams", type=str,help='Comma separated list of beams. Each beam is either a float for FWHM in arcminutes or the name of an array whose beam will be used.')
parser.add_argument("-N", "--nsims",     type=int,  default=1,help="A description.")
parser.add_argument("--skip-inpainting", action='store_true',help='Do not inpaint.')
parser.add_argument("--theory",     type=str,  default="none",help="A description.")
parser.add_argument("--fg-res-version", type=str,help='Version name for residual foreground powers.',default='fgres_v1')
parser.add_argument("--sim-version", type=str,help='Region name.',default='v6.2.0_calibrated_mask_version_padded_v1')
parser.add_argument("--mask-version", type=str,  default="padded_v1",help='Mask version')
parser.add_argument("-o", "--overwrite", action='store_true',help='Ignore existing version directory.')
parser.add_argument("-m", "--memory-intensive", action='store_true',help='Do not save FFTs to scratch disk. Can be faster, but very memory intensive.')
parser.add_argument("--uncalibrated", action='store_true',help='Do not use calibration factors.')
parser.add_argument("--signal-bin-width",     type=int,  default=pdefaults['signal_bin_width'],help="A description.")
parser.add_argument("--signal-interp-order",     type=int,  default=pdefaults['signal_interp_order'],help="A description.")
parser.add_argument("--delta-ell",     type=int,  default=pdefaults['delta_ell'],help="A description.")
parser.add_argument("--set-id",     type=int,  default=0,help="Sim set id.")
parser.add_argument("--rfit-bin-width",     type=int,  default=pdefaults['rfit_bin_width'],help="A description.")
parser.add_argument("--rfit-wnoise-width",     type=int,  default=pdefaults['rfit_wnoise_width'],help="A description.")
parser.add_argument("--rfit-lmin",     type=int,  default=pdefaults['rfit_lmin'],help="A description.")
parser.add_argument("--chunk-size",     type=int,  default=5000000,help="Chunk size.")
parser.add_argument("--maxval",     type=float,  default=700000,help="Maxval for covmat.")
parser.add_argument("--beam-version", type=str,  default=None,help='Mask version')
parser.add_argument("-e", "--effective-freq", action='store_true',help='Ignore bandpass files and use effective frequency.')
parser.add_argument("--unsanitized-beam", action='store_true',help='Do not sanitize beam.')
args = parser.parse_args()


# Generate each ACT and Planck sim and store kdiffs,kcoadd in memory

set_id = args.set_id
bandpasses = not(args.effective_freq)
gconfig = io.config_from_yaml("input/data.yml")
save_path = sints.dconfig['tilec']['save_path']
mask = sints.get_act_mr3_crosslinked_mask(args.region,
                                          version=args.mask_version,
                                          kind='binary_apod')
shape,wcs = mask.shape,mask.wcs
Ny,Nx = shape
modlmap = enmap.modlmap(shape,wcs)

ngen = {}
ngen['act_mr3'] = actnoise.NoiseGen(args.sim_version,model="act_mr3",extract_region=mask,ncache=1,verbose=True)
ngen['planck_hybrid'] = actnoise.NoiseGen(args.sim_version,model="planck_hybrid",extract_region=mask,ncache=1,verbose=True)


arrays = args.arrays.split(',')
narrays = len(arrays)
nsims = args.nsims
aspecs = tutils.ASpecs().get_specs

jsim = pipeline.JointSim(arrays,args.fg_res_version,bandpassed=bandpasses)

for sim_index in range(nsims):


    """
    MAKE SIMS
    """
    jsim.update_signal_index(sim_index,set_idx=set_id)

    pa3_cache = {} # This assumes there are at most 2 pa3 arrays in the input
    sim_splits = []
    for aindex in range(narrays):
        qid = arrays[aindex]
        dmname = sints.arrays(qid,'data_model')
        dm = sints.models[dmname](region=mask,calibrated=not(args.uncalibrated))
        patch = args.region

        if dm.name=='act_mr3':
            season,array1,array2 = sints.arrays(qid,'season'),sints.arrays(qid,'array'),sints.arrays(qid,'freq')
            arrayname = array1 + "_" + array2
        elif dm.name=='planck_hybrid':
            season,arrayname = None,sints.arrays(qid,'freq')


        with bench.show("signal"):
            # (npol,Ny,Nx)
            signal = jsim.compute_map(mask.shape,mask.wcs,qid,
                                      include_cmb=True,include_tsz=True,include_fgres=True)


        # Special treatment for pa3
        farray = arrayname.split('_')[0]
        if farray=='pa3':
            try:
                noise,ivars = pa3_cache[arrayname]
                genmap = False
            except:
                genmap = True
        else:
            genmap = True

        if genmap:
            # (ncomp,nsplits,npol,Ny,Nx)
            noise_seed = seed_tracker.get_noise_seed(set_id, sim_index, ngen[dmname].dm, season, patch, farray, None)
            fnoise,fivars = ngen[dmname].generate_sim(season=season,patch=patch,array=farray,seed=noise_seed,apply_ivar=False)
            print(fnoise.shape,fivars.shape)
            if farray=='pa3': 
                ind150 = dm.array_freqs['pa3'].index('pa3_f150')
                ind090 = dm.array_freqs['pa3'].index('pa3_f090')
                pa3_cache['pa3_f150'] = (fnoise[ind150].copy(),fivars[ind150].copy())
                pa3_cache['pa3_f090'] = (fnoise[ind090].copy(),fivars[ind090].copy())
                ind = dm.array_freqs['pa3'].index(arrayname)
            else:
                ind = 0
            noise = fnoise[ind]
            ivars = fivars[ind]

        splits = actnoise.apply_ivar_window(signal[None,None]+noise[None],ivars[None])
        fname = get_temp_split_fname(qid,set_id,sim_index)
        enmap.write_map(fname,splits)
        assert splits.shape[0]==1
        sim_splits.append(fname)

    
    """
    SAVE COV
    """
    print("Beginning covariance calculation...")
    ind_str = str(set_id).zfill(2)+"_"+str(sim_index).zfill(4)
    sim_version = "%s_%s" % (args.version,ind_str)
    with bench.show("sim cov"):
        pipeline.build_and_save_cov(args.arrays,args.region,sim_version,args.mask_version,
                                    args.signal_bin_width,args.signal_interp_order,args.delta_ell,
                                    args.rfit_wnoise_width,args.rfit_lmin,
                                    args.overwrite,args.memory_intensive,args.uncalibrated,
                                    sim_splits=sim_splits,skip_inpainting=args.skip_inpainting,
                                    theory_signal=args.theory,unsanitized_beam=args.unsanitized_beam)

    # delete split files
    for aindex in range(narrays):
        qid = arrays[aindex]
        fname = get_temp_split_fname(qid,set_id,sim_index)
        os.remove(fname)



    """
    SAVE ILC
    """
    print("done")
    ilc_version = "map_%s_%s" % (args.version,ind_str)
    with bench.show("sim ilc"):
        print("starting")
        pipeline.build_and_save_ilc(args.arrays,args.region,ilc_version,sim_version,args.beam_version,
                                    args.solutions,args.beams,args.chunk_size,
                                    args.effective_freq,args.overwrite,args.maxval,
                                    unsanitized_beam=args.unsanitized_beam)
