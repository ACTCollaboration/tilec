"""

We load each map sim
We cross-correlate the CMB map with the lensed CMB alms
We cross-correlate the Y map with the input Y alms

"""

from __future__ import print_function
from orphics import maps,io,cosmology,mpi,stats
from pixell import enmap,curvedsky
import numpy as np
import os,sys,shutil
from actsims import noise as actnoise
from actsims.util import seed_tracker
from soapack import interfaces as sints
from enlib import bench
from tilec import pipeline,utils as tutils
import healpy as hp
from actsims.util import seed_tracker
from szar import foregrounds as fgs
from datetime import datetime

def get_input(input_name,set_idx,sim_idx,shape,wcs):
    if input_name=='CMB':
        cmb_type = 'LensedUnabberatedCMB'
        #cmb_type = 'LensedCMB' # !!!!!!!!!!!!!
        signal_path = sints.dconfig['actsims']['signal_path']
        cmb_file   = os.path.join(signal_path, 'fullsky%s_alm_set%02d_%05d.fits' %(cmb_type, set_idx, sim_idx))
        alms = hp.fitsfunc.read_alm(cmb_file, hdu = 1)
    elif input_name=='tSZ':
        ellmax = 5101
        ells = np.arange(ellmax)
        cyy = fgs.power_y(ells)[None,None]
        cyy[0,0][ells<2] = 0
        comptony_seed = seed_tracker.get_fg_seed(set_idx, sim_idx, 'comptony')
        alms = curvedsky.rand_alm_healpy(cyy, seed = comptony_seed)
    omap = enmap.zeros((1,)+shape[-2:],wcs)
    omap = curvedsky.alm2map(np.complex128(alms),omap,spin=0)[0]
    return omap

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
name_map = {'CMB':'cmb','tSZ':'comptony','CIB':'cib'}


components = {}
input_names = []
for solution in args.solutions.split(','):
    components[solution] = solution.split('-')
    input_names.append( components[solution][0] )
input_names = set(input_names)

bin_edges = np.arange(80,5000,80)

s = stats.Stats(comm)

for i,task in enumerate(my_tasks):
    sim_index = task

    print("Rank %d starting task %d at %s..." % (rank,task,str(datetime.now())))


    # Get directory
    ind_str = str(set_id).zfill(2)+"_"+str(sim_index).zfill(4)
    version = "map_%s_%s" % (args.version,ind_str)
    savedir = tutils.get_save_path(version,args.region)
    assert os.path.exists(savedir)

    # Load mask
    if i==0:
        mask = enmap.read_map(savedir + "/tilec_mask.fits")
        shape,wcs = mask.shape,mask.wcs
        modlmap = mask.modlmap()
        binner = stats.bin2D(modlmap,bin_edges)
        w2 = np.mean(mask**2.)

    inputs = {}
    for input_name in input_names:
        inputs[input_name] = enmap.fft(get_input(input_name,args.set_id,sim_index,shape,wcs) * mask,normalize='phys')
        s.add_to_stats(input_name,binner.bin(np.real(inputs[input_name]*inputs[input_name].conj()))[1]/w2)

    for solution in args.solutions.split(','):
        comps = "tilec_single_tile_"+region+"_"
        comps = comps + name_map[components[solution][0]]+"_"
        if len(components[solution])>1: comps = comps + "deprojects_"+ '_'.join([name_map[x] for x in components[solution][1:]]) + "_"
        comps = comps + version    
        fname = "%s/%s.fits" % (savedir,comps)
        imap = enmap.read_map(fname)
        ls,bells = np.loadtxt("%s/%s_beam.txt" % (savedir,comps),unpack=True)
        kbeam = maps.interp(ls,bells)(modlmap)
        with np.errstate(divide='ignore',invalid='ignore'): kmap = enmap.fft(imap,normalize='phys')/kbeam
        kmap[~np.isfinite(kmap)] = 0
        s.add_to_stats("auto_"+solution,binner.bin(np.real(kmap*kmap.conj()))[1]/w2)
        input_name = components[solution][0]
        s.add_to_stats("cross_"+solution,binner.bin(np.real(inputs[input_name]*kmap.conj()))[1]/w2)


s.get_stats()

if rank==0:

    cents = binner.centers
    for input_name in input_names:
        pl = io.Plotter(xyscale='linlog',scalefn=lambda x: x**2./2./np.pi,xlabel='$\\ell$',ylabel='$D_{\\ell}$')
        ii = s.stats[input_name]['mean']
        pl.add(cents,ii,color='k',lw=3)
        for i,solution in enumerate(args.solutions.split(',')):
            color = "C%d" % i
            iname = components[solution][0]
            if iname!=input_name: continue
            ri = s.stats["cross_"+solution]['mean']
            pl.add(cents,ri,label=solution,ls="none",marker="o",color=color,markersize=2,alpha=0.8)

            rr = s.stats["auto_"+solution]['mean']
            pl.add(cents,rr,alpha=0.4,color=color)
        pl.done(os.environ['WORK']+"/val_%s.png" % input_name)


    for input_name in input_names:
        pl = io.Plotter(xyscale='linlin',xlabel='$\\ell$',ylabel='$\Delta C_{\\ell} / C_{\\ell}$')
        ii = s.stats[input_name]['mean']
        for i,solution in enumerate(args.solutions.split(',')):
            color = "C%d" % i
            iname = components[solution][0]
            if iname!=input_name: continue
            ri = s.stats["cross_"+solution]['mean']
            pl.add(cents,(ri-ii)/ii,label=solution,ls="none",marker="o",color=color,markersize=2,alpha=0.8)
            print(((ri-ii)/ii)[np.logical_and(cents>500,cents<1000)])
        pl.hline(y=0)
        pl._ax.set_ylim(-0.2,0.2)
        pl.vline(x=80)
        pl.vline(x=100)
        pl.vline(x=300)
        pl.vline(x=500)
        pl.vline(x=3000)
        pl.done(os.environ['WORK']+"/valdiff_%s.png" % input_name)
