"""
Tiled logic

- Have a list of array geometries in memory
- Loop through each array
- Check if center of tile falls in array geometry
- If not, skip array
- Extract full 8 degree tile from ivar of array
- if any pixel in central 4 degree of tile is missing (ivar=0), skip array
- Extract full 8 degree tile from map of array


We still need to apodize-mask the sharp edges of arrays that enter outside the central region of the tile.

"""
import os,sys
import numpy as np
from soapack import interfaces as sints
from pixell import enmap
from tilec import tiling
from orphics import mpi, io
comm = mpi.MPI.COMM_WORLD


def load_geometries(qids):
    geoms = {}
    for qid in qids:
        dmodel = sints.arrays(qid,'data_model')
        season = sints.arrays(qid,'season')
        region = sints.arrays(qid,'region')
        array = sints.arrays(qid,'array')
        freq = sints.arrays(qid,'freq')
        dm = sints.models[dmodel]()
        shape,wcs = enmap.read_map_geometry(dm.get_split_fname(season=season,patch=region,array=array+"_"+freq,splitnum=0,srcfree=True))
        geoms[qid] = shape[-2:],wcs 
    return geoms

def get_splits_ivar(qid,extracter):
    dmodel = sints.arrays(qid,'data_model')
    season = sints.arrays(qid,'season')
    region = sints.arrays(qid,'region')
    array = sints.arrays(qid,'array')
    freq = sints.arrays(qid,'freq')
    dm = sints.models[dmodel]()
    ivars = []
    for i in range(dm.get_nsplits(season=season,patch=region,array=array)):
        omap = extracter(dm.get_split_ivar_fname(season=season,patch=region,array=array+"_"+freq,splitnum=i))
        eshape,ewcs = omap.shape,omap.wcs
        ivars.append(omap.copy())
    return enmap.enmap(np.stack(ivars),ewcs)

def get_splits(qid,extracter):
    dmodel = sints.arrays(qid,'data_model')
    season = sints.arrays(qid,'season')
    region = sints.arrays(qid,'region')
    array = sints.arrays(qid,'array')
    freq = sints.arrays(qid,'freq')
    dm = sints.models[dmodel]()
    splits = []
    for i in range(dm.get_nsplits(season=season,patch=region,array=array)):
        omap = extracter(dm.get_split_fname(season=season,patch=region,array=array+"_"+freq,splitnum=i,srcfree=True),sel=np.s_[0,...]) # sel
        eshape,ewcs = omap.shape,omap.wcs
        splits.append(omap.copy())
    return enmap.enmap(np.stack(splits),ewcs)

def is_planck(qid):
    dmodel = sints.arrays(qid,'data_model')
    return True if dmodel=='planck_hybrid' else False
    
    
# args
qids = ['d5',
        'd6',
        'd56_01']
        # 'd56_02',
        # 'd56_03',
        # 'd56_04',
        # 'd56_05',
        # 'd56_06'] # list of quick IDs
parent_qid = 'd56_01' # qid of array whose geometry will be used for the full map

#qids = args.qids.split(',')
geoms = load_geometries(qids)
print(geoms)
pshape,pwcs = geoms[parent_qid]
ta = tiling.TiledAnalysis(pshape,pwcs,comm=comm,width_deg=4.,pix_arcmin=0.5)
ta.initialize_output(name="processed")

for extracter,inserter,eshape,ewcs in ta.tiles(from_file=True): # this is an MPI loop
    # What is the shape and wcs of the tile? is this needed?
    aids = []
    for qid in qids:
        # Check if this array is useful
        ashape,awcs = geoms[qid]
        Ny,Nx = ashape[-2:]
        center = enmap.center(eshape,ewcs)
        acpixy,acpixx = enmap.sky2pix(ashape,awcs,center)
        # Following can be made more restrictive by being aware of tile shape
        if acpixy<=0 or acpixx<=0 or acpixy>=Ny or acpixx>=Nx: continue
        # Ok so the center of the tile is inside this array, but are there any missing pixels?
        eivars = get_splits_ivar(qid,extracter)
        # Only check for missing pixels if array is not a Planck array
        if not(is_planck(qid)) and np.any(ta.crop_main(eivars)<=0): continue
        io.hplot(eivars)
        esplits = get_splits(qid,extracter)
        io.hplot(esplits*ta.apod)
        continue
        kspace.process
        aids.append(aid)
    #ilc.build_empirical_cov
    #pmap = ilc.do_ilc
    #ta.update_output("processed",pmap,inserter)






