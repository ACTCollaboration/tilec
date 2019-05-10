from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap
import numpy as np
import os,sys
from actsims import simgen
from soapack import interfaces as sints
from enlib import bench
from tilec import pipeline

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
parser.add_argument("--sim-version", type=str,help='Region name.',default='v5.3.0_mask_version_padded_v1')
parser.add_argument("-e", "--effective-freq", action='store_true',help='Ignore bandpass files and use effective frequency.')
parser.add_argument("--mask-version", type=str,  default="padded_v1",help='Mask version')
parser.add_argument("-o", "--overwrite", action='store_true',help='Ignore existing version directory.')
parser.add_argument("-m", "--memory-intensive", action='store_true',help='Do not save FFTs to scratch disk. Can be faster, but very memory intensive.')
parser.add_argument("--uncalibrated", action='store_true',help='Do not use calibration factors.')
parser.add_argument("--signal-bin-width",     type=int,  default=pdefaults['signal_bin_width'],help="A description.")
parser.add_argument("--signal-interp-order",     type=int,  default=pdefaults['signal_interp_order'],help="A description.")
parser.add_argument("--dfact",     type=int,  default=pdefaults['dfact'],help="A description.")
parser.add_argument("--rfit-bin-width",     type=int,  default=pdefaults['rfit_bin_width'],help="A description.")
parser.add_argument("--rfit-wnoise-width",     type=int,  default=pdefaults['rfit_wnoise_width'],help="A description.")
parser.add_argument("--rfit-lmin",     type=int,  default=pdefaults['rfit_lmin'],help="A description.")
args = parser.parse_args()


# Generate each ACT and Planck sim and store kdiffs,kcoadd in memory

simgen = simgen.SimGen(version=args.sim_version,max_cached=max_caches[args.region],cmb_type='LensedUnabberatedCMB')


bandpasses = not(args.effective_freq)
gconfig = io.config_from_yaml("input/data.yml")
save_path = sints.dconfig['tilec']['save_path']
mask = sints.get_act_mr3_crosslinked_mask(args.region,
                                          version=args.mask_version,
                                          kind='binary_apod')
shape,wcs = mask.shape,mask.wcs
Ny,Nx = shape
modlmap = enmap.modlmap(shape,wcs)

arrays = args.arrays.split(',')
narrays = len(arrays)
nsims = args.nsims

for sim_index in range(nsims):
    pa3_cache = {} # This assumes there are at most 2 pa3 arrays in the input
    sim_splits = []
    for aindex in range(narrays):
        array = arrays[aindex]
        ainfo = gconfig[array]
        dm = sints.models[ainfo['data_model']](region=mask)
        array_id = ainfo['id']
        if dm.name=='act_mr3':
            season,array1,array2 = array_id.split('_')
            narray = array1 + "_" + array2
            patch = args.region
        elif dm.name=='planck_hybrid':
            season,patch,narray = None,None,array_id

        # Special treatment for pa3
        farray = narray.split('_')[0]
        if farray=='pa3':
            freq = narray.split('_')[1]
            try: 
                imap = pa3_cache[narray]
                gen_map = False
            except:
                gen_map = True
        else:
            gen_map = True

        # Generate sim if necessary
        if gen_map:
            with bench.show("sim"):
                imap = simgen.get_sim(season, patch, farray,sim_num=sim_index)[:,:,0,...] # only intensity
               
        # Decide which map to use for pa3
        if farray=='pa3':
            pa3_cache[narray] = imap
            findex = dm.array_freqs[farray].index(narray)     
        else:
            findex = 0
        
        splits = imap[findex]
        print(splits.shape)
        sim_splits.append(splits.copy())

    
    ind_str = str(sim_index).zfill(int(np.log10(nsims))+2)
    sim_version = "%s_%s" % (args.version,ind_str)
    pipeline.build_and_save_cov(args.arrays,args.region,sim_version,args.mask_version,
                                args.signal_bin_width,args.signal_interp_order,args.dfact,
                                args.rfit_wnoise_width,args.rfit_lmin,
                                args.overwrite,args.memory_intensive,args.uncalibrated,
                                sim_splits=sim_splits)

