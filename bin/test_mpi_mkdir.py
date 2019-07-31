from __future__ import print_function
from orphics import maps,io,cosmology,mpi
from pixell import enmap
import numpy as np
import os,sys,shutil
from actsims import noise as actnoise
from actsims.util import seed_tracker
from soapack import interfaces as sints
from enlib import bench
from tilec import pipeline,utils as tutils

def sim_build(region,version,overwrite):

    save_scratch = True
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

    #enmap.write_map(scratch+"/temp1.hdf",enmap.enmap(np.zeros((100,100))))
    #enmap.write_map(savedir+"/temp2.hdf",enmap.enmap(np.zeros((100,100))))

    np.save(scratch+"/temp1.npy",enmap.enmap(np.zeros((100,100))))
    np.save(savedir+"/temp2.npy",enmap.enmap(np.zeros((100,100))))

    shutil.rmtree(scratch)


def sim_ilc(region,version,cov_version,overwrite):

    savedir = tutils.get_save_path(version,region)
    covdir = tutils.get_save_path(cov_version,region)
    print(covdir)
    assert os.path.exists(covdir)
    if not(overwrite):
        assert not(os.path.exists(savedir)), \
       "This version already exists on disk. Please use a different version identifier."
    try: os.makedirs(savedir)
    except:
        if overwrite: pass
        else: raise


    #enmap.write_map(savedir+"/temp2.hdf",enmap.enmap(np.zeros((100,100))))
    np.save(savedir+"/temp2.npy",enmap.enmap(np.zeros((100,100))))


import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("version", type=str,help='Version name.')
parser.add_argument("region", type=str,help='Region name.')
parser.add_argument("-N", "--nsims",     type=int,  default=1,help="A description.")
parser.add_argument("-o", "--overwrite", action='store_true',help='Ignore existing version directory.')
parser.add_argument("--set-id",     type=int,  default=0,help="Sim set id.")
args = parser.parse_args()

# Generate each ACT and Planck sim and store kdiffs,kcoadd in memory
nsims = args.nsims
set_id = args.set_id
comm,rank,my_tasks = mpi.distribute(nsims)

for task in my_tasks:
    sim_index = task

    ind_str = str(set_id).zfill(2)+"_"+str(sim_index).zfill(4)
    sim_version = "%s_%s" % (args.version,ind_str)
    scratch = tutils.get_scratch_path(sim_version,args.region)
    try: 
        os.makedirs(scratch)
    except: 
        pass
    """
    MAKE SIMS
    """

    
    """
    SAVE COV
    """
    print("Beginning covariance calculation...")
    with bench.show("sim cov"):
        sim_build(args.region,sim_version,args.overwrite)




    """
    SAVE ILC
    """
    print("done")
    ilc_version = "map_%s_%s" % (args.version,ind_str)
    with bench.show("sim ilc"):
        print("starting")
        sim_ilc(args.region,ilc_version,sim_version,args.overwrite)

    savepath = tutils.get_save_path(sim_version,args.region)
    shutil.rmtree(savepath)



