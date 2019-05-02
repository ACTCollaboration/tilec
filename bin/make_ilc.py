from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
from enlib import bench
import numpy as np
import os,sys
from tilec import fg as tfg,ilc
from soapack import interfaces as sints

"""
This script will work with a saved covariance matrix to obtain component separated
maps.

The datamodel is only used for beams here.
"""

chunk_size = 5000000
#chunk_size = 10000 # whether nans appear depends on chunk size!?
print("Chunk size is ", chunk_size*64./8./1024./1024./1024., " GB.")
def warn(): print("WARNING: no bandpass file found. Assuming array ",dm.c['id']," has no response to CMB, tSZ and CIB.")


import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("version", type=str,help='Region name.')
parser.add_argument("cov_version", type=str,help='Region name.')
parser.add_argument("region", type=str,help='Region name.')
parser.add_argument("arrays", type=str,help='Comma separated list of array names. Array names map to a data specification in data.yml')
parser.add_argument("solutions", type=str,help='Comma separated list of solutions. Each solution is of the form x-y-... where x is solved for and the optionally provided y-,... are deprojected. The x can belong to any of CMB,tSZ and y,z,... can belong to any of CMB,tSZ,CIB.')
parser.add_argument("beams", type=str,help='Comma separated list of beams. Each beam is either a float for FWHM in arcminutes or the name of an array whose beam will be used.')
parser.add_argument("-o", "--overwrite", action='store_true',help='Ignore existing version directory.')
parser.add_argument("-e", "--effective-freq", action='store_true',help='Ignore bandpass files and use effective frequency.')
parser.add_argument("--beam-version", type=str,  default=None,help='Mask version')
args = parser.parse_args()

bandpasses = not(args.effective_freq)
gconfig = io.config_from_yaml("input/data.yml")
save_path = sints.dconfig['tilec']['save_path']
savedir = save_path + args.version + "/" + args.region +"/"
covdir = save_path + args.cov_version + "/" + args.region +"/"
assert os.path.exists(covdir)
if not(args.overwrite):
    assert not(os.path.exists(savedir)), \
   "This version already exists on disk. Please use a different version identifier."
try: os.makedirs(savedir)
except:
    if args.overwrite: pass
    else: raise


mask = enmap.read_map(covdir+"tilec_mask.fits")
shape,wcs = mask.shape,mask.wcs
Ny,Nx = shape
modlmap = enmap.modlmap(shape,wcs)

arrays = args.arrays.split(',')
narrays = len(arrays)
kcoadds = []
kbeams = []
bps = []
names = []
for i,array in enumerate(arrays):
    ainfo = gconfig[array]
    dm = sints.models[ainfo['data_model']](region=mask)
    array_id = ainfo['id']
    names.append(array_id)
    if dm.name=='act_mr3':
        season,array1,array2 = array_id.split('_')
        narray = array1 + "_" + array2
        patch = args.region
    elif dm.name=='planck_hybrid':
        season,patch,narray = None,None,array_id
    kcoadd_name = covdir + "kcoadd_%s.hdf" % array
    ainfo = gconfig[array]
    lmax = ainfo['lmax']
    lmin = ainfo['lmin']
    kmask = maps.mask_kspace(shape,wcs,lmin=lmin,lmax=lmax)
    kcoadd = enmap.read_map(kcoadd_name)
    dtype = kcoadd.dtype
    kcoadds.append(kcoadd.copy()*kmask)
    kbeams.append(dm.get_beam(ells=modlmap,season=season,patch=patch,array=narray,version=args.beam_version))
    if bandpasses:
        try: bps.append("data/"+ainfo['bandpass_file'] )
        except:
            warn()
            bps.append(None)
    else:
        try: bps.append(ainfo['bandpass_approx_freq'])
        except:
            warn()
            bps.append(None)
                
kcoadds = enmap.enmap(np.stack(kcoadds),wcs)

# Read Covmat
maxval = np.loadtxt(covdir + "maximum_value_of_covariances.txt")
assert maxval.size==1
maxval = maxval.reshape(-1)[0]
cov = maps.SymMat(narrays,shape[-2:])
for aindex1 in range(narrays):
    for aindex2 in range(aindex1,narrays):
        icov = enmap.read_map(covdir+"tilec_hybrid_covariance_%s_%s.hdf" % (names[aindex1],names[aindex2]))
        # if aindex1==aindex2: # why is this necsessary if make_cov should already be doing it?
        #     # and should we do it for cross-covariances too?
        #     array = arrays[aindex1]
        #     ainfo = gconfig[array]
        #     lmax = ainfo['lmax']
        #     lmin = ainfo['lmin']
        #     icov[modlmap>=lmax] = 1e4 * maxval
        #     icov[modlmap<=lmin] = 1e4 * maxval
        cov[aindex1,aindex2] = icov
            
cov.data = enmap.enmap(cov.data,wcs,copy=False)
covfunc = lambda sel: cov.to_array(sel,flatten=True)

assert cov.data.shape[0]==((narrays*(narrays+1))/2) # FIXME: generalize
if np.any(np.isnan(cov.data)): raise ValueError 

# Make responses
responses = {}
for comp in ['tSZ','CMB','CIB']:
    if bandpasses:
        responses[comp] = tfg.get_mix_bandpassed(bps, comp)
    else:
        responses[comp] = tfg.get_mix(bps, comp)
        
ilcgen = ilc.chunked_ilc(modlmap,np.stack(kbeams),covfunc,chunk_size,responses=responses,invert=True)

# Initialize containers
solutions = args.solutions.split(',')
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
beams = args.beams.split(',')
for solution,beam in zip(solutions,beams):
    #comps = '_'.join(data[solution]['comps'])
    comps = "tilec_single_tile_"+args.region+"_"
    comps = comps + name_map[data[solution]['comps'][0]]+"_"
    if len(data[solution]['comps'])>1: comps = comps + "deprojects_"+ '_'.join([name_map[x] for x in data[solution]['comps'][1:]]) + "_"
    comps = comps + args.version
    try:
        noise = enmap.enmap(data[solution]['noise'].reshape((Ny,Nx)),wcs)
        enmap.write_map("%s/%s_noise.fits" % (savedir,comps),noise)
    except: pass
    try:
        cnoise = enmap.enmap(data[solution]['cnoise'].reshape((Ny,Nx)),wcs)
        enmap.write_map("%s/%s_cross_noise.fits" % (savedir,comps),noise)
    except: pass

    ells = np.arange(0,modlmap.max(),1)
    try:
        fbeam = float(beam)
        kbeam = maps.gauss_beam(modlmap,fbeam)
        lbeam = maps.gauss_beam(ells,fbeam)
    except:
        array = beam
        ainfo = gconfig[array]
        array_id = ainfo['id']
        dm = sints.models[ainfo['data_model']](region=mask)
        if dm.name=='act_mr3':
            season,array1,array2 = array_id.split('_')
            narray = array1 + "_" + array2
            patch = args.region
        elif dm.name=='planck_hybrid':
            season,patch,narray = None,None,array_id
        bfunc = lambda x: dm.get_beam(x,season=season,patch=patch,array=narray,version=args.beam_version)
        kbeam = bfunc(modlmap)
        lbeam = bfunc(ells)
        
    smap = enmap.ifft(kbeam*enmap.enmap(data[solution]['kmap'].reshape((Ny,Nx)),wcs),normalize='phys').real
    enmap.write_map("%s/%s.fits" % (savedir,comps),smap)
    io.save_cols("%s/%s_beam.txt" % (savedir,comps),(ells,lbeam),header="ell beam")
    

enmap.write_map(savedir+"/tilec_mask.fits",mask)
