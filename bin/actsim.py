from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap
import numpy as np
import os,sys
from actsims import noise
from soapack import interfaces as sints
from enlib import bench
from tilec import pipeline,utils as tutils

max_caches = {'deep56':1,'boss':1,'deep6':1}
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
parser.add_argument("--rfit-bin-width",     type=int,  default=pdefaults['rfit_bin_width'],help="A description.")
parser.add_argument("--rfit-wnoise-width",     type=int,  default=pdefaults['rfit_wnoise_width'],help="A description.")
parser.add_argument("--rfit-lmin",     type=int,  default=pdefaults['rfit_lmin'],help="A description.")
parser.add_argument("--chunk-size",     type=int,  default=5000000,help="Chunk size.")
parser.add_argument("--maxval",     type=float,  default=700000,help="Maxval for covmat.")
parser.add_argument("--beam-version", type=str,  default=None,help='Mask version')
parser.add_argument("-e", "--effective-freq", action='store_true',help='Ignore bandpass files and use effective frequency.')
args = parser.parse_args()


# Generate each ACT and Planck sim and store kdiffs,kcoadd in memory


bandpasses = not(args.effective_freq)
gconfig = io.config_from_yaml("input/data.yml")
save_path = sints.dconfig['tilec']['save_path']
mask = sints.get_act_mr3_crosslinked_mask(args.region,
                                          version=args.mask_version,
                                          kind='binary_apod')
shape,wcs = mask.shape,mask.wcs
Ny,Nx = shape
modlmap = enmap.modlmap(shape,wcs)


angen = noise.NoiseGen(args.sim_version,model="act_mr3",extract_region=mask,ncache=1,verbose=True)
pngen = noise.NoiseGen(args.sim_version,model="planck_hybrid",extract_region=mask,ncache=1,verbose=True)


arrays = args.arrays.split(',')
narrays = len(arrays)
nsims = args.nsims
aspecs = tutils.ASpecs().get_specs

jsim = pipeline.JointTemperatureSim(arrays,args.fg_res_version,bandpassed=bandpasses)

for sim_index in range(nsims):


    jsim.update_signal_index(sim_index)


    """
    LOAD DATA
    """
    pa3_cache = {} # This assumes there are at most 2 pa3 arrays in the input
    sim_splits = []
    for aindex in range(narrays):
        qid = arrays[aindex]
        dm = sints.models[sints.arrays(qid,'data_model')](region=mask,calibrated=not(args.uncalibrated))
        patch = args.region
        if dm.name=='act_mr3':
            season,array1,array2 = sints.arrays(qid,'season'),sints.arrays(qid,'array'),sints.arrays(qid,'freq')
            arrayname = array1 + "_" + array2
            simgen = asimgen
        elif dm.name=='planck_hybrid':
            season,arrayname = None,sints.arrays(qid,'freq')
            simgen = psimgen

        # Special treatment for pa3
        farray = arrayname.split('_')[0]
        if farray=='pa3':
            freq = arrayname.split('_')[1]
            try: 
                imap = pa3_cache[arrayname]
                gen_map = False
            except:
                gen_map = True
        else:
            gen_map = True

        # Generate sim if necessary
        if gen_map:
            with bench.show("sim"):
                imap = simgen.get_sim(season, patch, farray,sim_num=sim_index) #[:,:,0,...] # only intensity
               
        # Decide which map to use for pa3
        if farray=='pa3':
            pa3_cache[arrayname] = imap
            findex = dm.array_freqs[farray].index(arrayname)
        else:
            findex = 0
        
        splits = imap[findex]
        print(splits.shape)
        sim_splits.append(splits.copy())

    
    """
    SAVE COV
    """
    ind_str = str(sim_index).zfill(int(np.log10(nsims))+2)
    sim_version = "%s_%s" % (args.version,ind_str)
    with bench.show("sim cov"):
        pipeline.build_and_save_cov(args.arrays,args.region,sim_version,args.mask_version,
                                    args.signal_bin_width,args.signal_interp_order,args.delta_ell,
                                    args.rfit_wnoise_width,args.rfit_lmin,
                                    args.overwrite,args.memory_intensive,args.uncalibrated,
                                    sim_splits=sim_splits,skip_inpainting=args.skip_inpainting,theory_signal=args.theory)


    """
    SAVE ILC
    """
    print("done")
    ilc_version = "map_%s_%s" % (args.version,ind_str)
    with bench.show("sim ilc"):
        print("starting")
        pipeline.build_and_save_ilc(args.arrays,args.region,ilc_version,sim_version,args.beam_version,
                                    args.solutions,args.beams,args.chunk_size,
                                    args.effective_freq,args.overwrite,args.maxval)
