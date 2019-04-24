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
"""

chunk_size = 1000000
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
parser.add_argument("--mask-version", type=str,  default="padded_v1",help='Mask version')
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


mask = sints.get_act_mr3_crosslinked_mask(args.region,
                                          version=args.mask_version,
                                          kind='binary_apod')
shape,wcs = mask.shape,mask.wcs
Ny,Nx = shape
modlmap = enmap.modlmap(shape,wcs)

arrays = args.arrays.split(',')
narrays = len(arrays)
with bench.show("ffts"):
    kcoadds = []
    kbeams = []
    bps = []
    for i,array in enumerate(arrays):
        ainfo = gconfig[array]
        dm = sints.models[ainfo['data_model']](region=mask)
        array_id = ainfo['id']
        if dm.name=='act_mr3':
            season,array1,array2 = array_id.split('_')
            narray = array1 + "_" + array2
            patch = args.region
        elif dm.name=='planck_hybrid':
            season,patch,narray = None,None,array_id
        kcoadd_name = covdir + "kcoadd_%s.hdf" % array
        kcoadd = enmap.read_map(kcoadd_name)
        kcoadds.append(kcoadd.copy())
        kbeams.append(dm.get_beam(ells=modlmap,season=season,patch=patch,array=narray))
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
cov = enmap.read_map("%s/datacov_triangle.hdf" % covdir)
cov = maps.symmat_from_data(cov)
assert cov.data.shape[0]==((narrays*(narrays+1))/2) # FIXME: generalize
if np.any(np.isnan(cov.data)): raise ValueError 

# Make responses
responses = {}
for comp in ['tSZ','CMB','CIB']:
    if bandpasses:
        responses[comp] = tfg.get_mix_bandpassed(bps, comp)
    else:
        responses[comp] = tfg.get_mix(bps, comp)
        
ilcgen = ilc.chunked_ilc(modlmap,np.stack(kbeams),cov,chunk_size,responses=responses,invert=True)

# Initialize containers
solutions = args.solutions.split(',')
data = {}
kcoadds = kcoadds.reshape((narrays,Ny*Nx))
for solution in solutions:
    data[solution] = {}
    comps = solution.split('-')
    data[solution]['comps'] = comps
    if len(comps)<=2: data[solution]['noise'] = enmap.empty((Ny*Nx),wcs)
    data[solution]['kmap'] = enmap.empty((Ny*Nx),wcs,dtype=np.complex128) # FIXME: reduce dtype?
        
for chunknum,(hilc,selchunk) in enumerate(ilcgen):
    print("ILC on chunk ", chunknum+1, " / ",int(modlmap.size/chunk_size)+1," ...")
    for solution in solutions:
        comps = data[solution]['comps']
        if len(comps)==1: # GENERALIZE
            data[solution]['noise'][selchunk] = hilc.standard_noise(comps[0])
            data[solution]['kmap'][selchunk] = hilc.standard_map(kcoadds[...,selchunk],comps[0])
        elif len(comps)==2:
            data[solution]['noise'][selchunk] = hilc.constrained_noise(comps[0],comps[1])
            data[solution]['kmap'][selchunk] = hilc.constrained_map(kcoadds[...,selchunk],comps[0],comps[1])
        elif len(comps)>2:
            data[solution]['kmap'][selchunk] = np.nan_to_num(hilc.multi_constrained_map(kcoadds[...,selchunk],comps[0],*comps[1:]))

del ilcgen,cov

# Reshape into maps
beams = args.beams.split(',')
for solution,beam in zip(solutions,beams):
    comps = '_'.join(data[solution]['comps'])
    try:
        noise = enmap.enmap(data[solution]['noise'].reshape((Ny,Nx)),wcs)
        enmap.write_map("%s/%s_noise.fits" % (savedir,comps),noise)
    except: pass

    ells = np.arange(0,modlmap.max(),1)
    try:
        fbeam = float(beam)
        kbeam = maps.gauss_beam(modlmap,fbeam)
        lbeam = maps.gauss_beam(ells,fbeam)
    except:
        array = beam
        array_datamodel = gconfig[array]['data_model']
        dm = datamodel.datamodels[array_datamodel](args.region,gconfig[array])
        kbeam = dm.get_beam(modlmap)
        lbeam = dm.get_beam(ells)
        
    smap = enmap.ifft(kbeam*enmap.enmap(data[solution]['kmap'].reshape((Ny,Nx)),wcs),normalize='phys').real
    enmap.write_map("%s/%s_kmap.fits" % (savedir,comps),smap)
    io.save_cols("%s/%s_beam.txt" % (savedir,comps),(ells,lbeam))
    

