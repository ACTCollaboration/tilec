from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap
import numpy as np
import os,sys
from actsims import simgen
from soapack import interfaces as sints
from enlib import bench

max_caches = {'deep56':1,'boss':1}

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("region", type=str,help='Region name.')
parser.add_argument("arrays", type=str,help='Comma separated list of array names. Array names map to a data specification in data.yml')
parser.add_argument("solutions", type=str,help='Comma separated list of solutions. Each solution is of the form x-y-... where x is solved for and the optionally provided y-,... are deprojected. The x can belong to any of CMB,tSZ and y,z,... can belong to any of CMB,tSZ,CIB.')
parser.add_argument("beams", type=str,help='Comma separated list of beams. Each beam is either a float for FWHM in arcminutes or the name of an array whose beam will be used.')
parser.add_argument("-N", "--nsim",     type=int,  default=1,help="A description.")
parser.add_argument("--version", type=str,help='Region name.',default='v5.2.1_mask_version_padded_v1')
parser.add_argument("-e", "--effective-freq", action='store_true',help='Ignore bandpass files and use effective frequency.')
parser.add_argument("--mask-version", type=str,  default="padded_v1",help='Mask version')
args = parser.parse_args()


# Generate each ACT and Planck sim and store kdiffs,kcoadd in memory

simgen = simgen.SimGen(version=args.version,max_cached=max_caches[args.region],cmb_type='LensedUnabberatedCMB')


bandpasses = not(args.effective_freq)
gconfig = io.config_from_yaml("input/data.yml")
save_path = sints.dconfig['tilec']['save_path']
savedir = save_path + args.version + "/" + args.region +"/"

mask = sints.get_act_mr3_crosslinked_mask(args.region,
                                          version=args.mask_version,
                                          kind='binary_apod')
shape,wcs = mask.shape,mask.wcs
Ny,Nx = shape
modlmap = enmap.modlmap(shape,wcs)

arrays = args.arrays.split(',')
narrays = len(arrays)
nsims = args.nsim

for sim_index in range(nsims):
    pa3_cache = {}
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
               
        print(imap.shape)
        # Decide which map to use for pa3
        if farray=='pa3':
            pa3_cache[narray] = imap
            findex = dm.array_freqs[farray].index(narray)     
        else:
            findex = 0
        
        splits = imap[findex]

        print(splits.shape)

    

    

# 
