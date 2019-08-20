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

def get_input(input_name,set_idx,sim_idx,shape,wcs):
    if input_name=='CMB':
        cmb_type = 'LensedUnabberatedCMB'
        #cmb_type = 'LensedCMB' # !!!!!!!!!!!!!
        signal_path = sints.dconfig['actsims']['signal_path']
        cmb_file   = os.path.join(signal_path, 'fullsky%s_alm_set%02d_%05d.fits' %(cmb_type, set_idx, sim_idx))
        alms = hp.fitsfunc.read_alm(cmb_file, hdu = 1)
    elif input_name=='tSZ':
        ellmax = 8101
        ells = np.arange(ellmax)
        cyy = fgs.power_y(ells)[None,None]
        cyy[0,0][ells<2] = 0
        comptony_seed = seed_tracker.get_fg_seed(set_idx, sim_idx, 'comptony')
        alms = curvedsky.rand_alm(cyy, seed = comptony_seed,lmax=ellmax)
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
input_names = sorted(list(set(input_names)))

bin_edges = np.arange(80,8000,80)

#s = stats.Stats(comm)

totinputs = {}
totautos = {}
totcrosses = {}

for i,task in enumerate(my_tasks):
    sim_index = task

    print("Rank %d starting task %d at %s..." % (rank,task,str(datetime.now())))


    # Get directory
    ind_str = str(set_id).zfill(2)+"_"+str(sim_index).zfill(4)
    #ind_str = str(set_id).zfill(2)+"_"+str(sim_index*2).zfill(4) # !!! 
    version = "map_joint_%s_%s" % (args.version,ind_str)
    savedir = tutils.get_save_path(version,args.region)
    print(savedir)
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
        #inputs[input_name] = enmap.fft(get_input(input_name,args.set_id,sim_index*2,shape,wcs) * mask,normalize='phys') # !!!
        res = binner.bin(np.real(inputs[input_name]*inputs[input_name].conj()))[1]/w2
        print(rank,task,input_name,res[10])
        try: totinputs[input_name] = totinputs[input_name] + res
        except: totinputs[input_name] = res.copy()

    for solution in args.solutions.split(','):
        comps = "tilec_single_tile_"+region+"_"
        comps = comps + name_map[components[solution][0]]+"_"
        if len(components[solution])>1: comps = comps + "deprojects_"+ '_'.join([name_map[x] for x in components[solution][1:]]) + "_"
        comps = comps + version    
        fname = "%s/%s.fits" % (savedir,comps)
        imap = enmap.read_map(fname)
        ls,bells = np.loadtxt("%s/%s_beam.txt" % (savedir,comps),unpack=True)
        kbeam = maps.interp(ls,bells)(modlmap)

        # rbeam = maps.gauss_beam(imap.modlmap(),10.)/kbeam
        # rbeam[~np.isfinite(rbeam)]=0
        # if task==0: io.hplot(maps.filter_map(imap,rbeam),"/scratch/r/rbond/msyriac/cmap")# !!!
        # sys.exit()

        with np.errstate(divide='ignore',invalid='ignore'): kmap = enmap.fft(imap,normalize='phys')/kbeam
        kmap[~np.isfinite(kmap)] = 0
        res = binner.bin(np.real(kmap*kmap.conj()))[1]/w2
        try: totautos[solution] = totautos[solution] + res
        except: totautos[solution] = res.copy()
        input_name = components[solution][0]
        res = binner.bin(np.real(inputs[input_name]*kmap.conj()))[1]/w2
        #s.add_to_stats("cross_"+solution,res)
        try: totcrosses[solution] = totcrosses[solution] + res
        except: totcrosses[solution] = res.copy()


#s.get_stats()
# totcmb = putils.allreduce(totcmb,comm) /nsims 
#tottsz = putils.allreduce(tottsz,comm) /nsims 
for key in sorted(totinputs.keys()):
    totinputs[key] = putils.allreduce(totinputs[key],comm) /nsims
for key in sorted(totautos.keys()):
    totautos[key] = putils.allreduce(totautos[key],comm) /nsims
for key in sorted(totcrosses.keys()):
    totcrosses[key] = putils.allreduce(totcrosses[key],comm) /nsims

if rank==0:

    cents = binner.centers
    for input_name in input_names:
        pl = io.Plotter(xyscale='linlog',scalefn=lambda x: x**2./2./np.pi,xlabel='$\\ell$',ylabel='$D_{\\ell}$')
        # ii = totcmb if input_name=='CMB' else tottsz
        ii = totinputs[input_name]
        pl.add(cents,ii,color='k',lw=3)
        for i,solution in enumerate(args.solutions.split(',')):
            color = "C%d" % i
            iname = components[solution][0]
            if iname!=input_name: continue
            ri = totcrosses[solution]
            pl.add(cents,ri,label=solution,ls="none",marker="o",color=color,markersize=2,alpha=0.8)

            rr = totautos[solution]
            pl.add(cents,rr,alpha=0.4,color=color)
        pl.done(os.environ['WORK']+"/val_%s.png" % input_name)


    for input_name in input_names:
        pl = io.Plotter(xyscale='linlin',xlabel='$\\ell$',ylabel='$\Delta C_{\\ell} / C_{\\ell}$')
        # ii = totcmb if input_name=='CMB' else tottsz
        ii = totinputs[input_name]
        for i,solution in enumerate(args.solutions.split(',')):
            color = "C%d" % i
            iname = components[solution][0]
            if iname!=input_name: continue
            ri = totcrosses[solution]
            pl.add(cents,(ri-ii)/ii,label=solution,ls="none",marker="o",color=color,markersize=2,alpha=0.8)
            print(((ri-ii)/ii)[np.logical_and(cents>500,cents<1000)])
        pl.hline(y=0)
        pl._ax.set_ylim(-0.1,0.1)
        pl.vline(x=80)
        pl.vline(x=100)
        pl.vline(x=300)
        pl.vline(x=500)
        pl.vline(x=3000)
        pl.done(os.environ['WORK']+"/valdiff_%s.png" % input_name)
