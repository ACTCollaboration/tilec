"""

We load each map sim
We cross-correlate the CMB map with the lensed CMB alms
We cross-correlate the Y map with the input Y alms

"""

from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from orphics import maps,io,cosmology,mpi,stats
from pixell import enmap,curvedsky,utils as putils
import numpy as np
import os,sys,shutil
from actsims.util import seed_tracker
from soapack import interfaces as sints
from enlib import bench
from tilec import pipeline,utils as tutils
import healpy as hp
from szar import foregrounds as fgs
from datetime import datetime

# def get_input(input_name,set_idx,sim_idx,shape,wcs):
#     if input_name=='CMB':
#         cmb_type = 'LensedUnabberatedCMB'
#         #cmb_type = 'LensedCMB' # !!!!!!!!!!!!!
#         signal_path = sints.dconfig['actsims']['signal_path']
#         cmb_file   = os.path.join(signal_path, 'fullsky%s_alm_set%02d_%05d.fits' %(cmb_type, set_idx, sim_idx))
#         alms = hp.fitsfunc.read_alm(cmb_file, hdu = 1)
#     elif input_name=='tSZ':
#         ellmax = 5101
#         ells = np.arange(ellmax)
#         cyy = fgs.power_y(ells)[None,None]
#         cyy[0,0][ells<2] = 0
#         comptony_seed = seed_tracker.get_fg_seed(set_idx, sim_idx, 'comptony')
#         alms = curvedsky.rand_alm_healpy(cyy, seed = comptony_seed)
#     omap = enmap.zeros((1,)+shape[-2:],wcs)
#     omap = curvedsky.alm2map(np.complex128(alms),omap,spin=0)[0]
#     return omap

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("version", type=str,help='Version name.')
parser.add_argument("region", type=str,help='Region name.')
parser.add_argument("solutions", type=str,help='Comma separated list of solutions. Each solution is of the form x-y-... where x is solved for and the optionally provided y-,... are deprojected. The x can belong to any of CMB,tSZ and y,z,... can belong to any of CMB,tSZ,CIB.')
parser.add_argument("-N", "--nsims",     type=int,  default=1,help="A description.")
parser.add_argument("--set-id",     type=int,  default=0,help="Sim set id.")
args = parser.parse_args()

region = args.region
nsims = args.nsims
set_id = args.set_id
comm,rank,my_tasks = mpi.distribute(nsims)


components = {}
input_names = []
for solution in args.solutions.split(','):
    components[solution] = solution.split('-')
    input_names.append( components[solution][0] )
input_names = sorted(list(set(input_names)))
#input_names = ['tSZ','CMB'] #list(set(input_names))
print(input_names)

bin_edges = np.arange(80,5000,80)

totinputs = {}

for i,task in enumerate(my_tasks):
    sim_index = task

    print("Rank %d starting task %d at %s..." % (rank,task,str(datetime.now())))

    # Get directory
    # ind_str = str(set_id).zfill(2)+"_"+str(sim_index).zfill(4)
    # version = "map_%s_%s" % (args.version,ind_str)
    # savedir = tutils.get_save_path(version,args.region)
    # assert os.path.exists(savedir)

    # Load mask
    # if i==0:
    #     mask = enmap.read_map(savedir + "/tilec_mask.fits")
    #     shape,wcs = mask.shape,mask.wcs
    #     modlmap = mask.modlmap()
    #     binner = stats.bin2D(modlmap,bin_edges)
    #     w2 = np.mean(mask**2.)

    inputs = {}
    for input_name in input_names:
        # kmap= enmap.fft(get_input(input_name,args.set_id,sim_index,shape,wcs) * mask,normalize='phys')
        # res = binner.bin(np.real(kmap*kmap.conj()))[1]/w2
        res = np.ones((5,))*1e14 if input_name=='CMB' else np.ones((5,))*1e-14 #!!!!!
        print(rank,task,input_name,res[3])
        try: totinputs[input_name] = totinputs[input_name] + res
        except: totinputs[input_name] = res.copy()

for key in sorted(totinputs.keys()):
    totinputs[key] = putils.allreduce(totinputs[key],comm) /nsims

if rank==0:

    # cents = binner.centers
    for input_name in input_names:
        ii = totinputs[input_name]
        print(input_name,ii)
