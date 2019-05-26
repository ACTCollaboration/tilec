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

def apodize_zero(imap,width):
    ivar = imap.copy()
    ivar[...,:1,:] = 0; ivar[...,-1:] = 0; ivar[...,:,:1] = 0; ivar[...,:,-1:] = 0
    from scipy import ndimage
    dist = ndimage.distance_transform_edt(ivar>0)
    apod = 0.5*(1-np.cos(np.pi*np.minimum(1,dist/width)))
    return apod

def is_planck(qid):
    dmodel = sints.arrays(qid,'data_model')
    return True if dmodel=='planck_hybrid' else False
    
def coadd(imap,ivar):
    isum = np.sum(ivar,axis=0)
    c = np.sum(imap*ivar,axis=0)/isum
    c[~np.isfinite(c)] = 0
    return c,isum
    
# args
qids = ['d5',
        'd6',
        'd56_01',
        'd56_02',
        'd56_03',
        'd56_04',
        'd56_05',
        'd56_06','s16_01','s16_02','s16_03','p04','p05'] # list of quick IDs
parent_qid = 'd56_01' # qid of array whose geometry will be used for the full map

#qids = args.qids.split(',')
geoms = load_geometries(qids)
pshape,pwcs = geoms[parent_qid]
ta = tiling.TiledAnalysis(pshape,pwcs,comm=comm,width_deg=4.,pix_arcmin=0.5)
ta.initialize_output(name="processed")
ta.initialize_output(name="processed_ivar")
down = lambda x,n=2: enmap.downgrade(x,n)

for i,extracter,inserter,eshape,ewcs in ta.tiles(from_file=True): # this is an MPI loop
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
        if not(is_planck(qid)) and np.any(ta.crop_main(eivars)<=0):
            continue

        apod = ta.apod * apodize_zero(np.sum(eivars,axis=0),60) #eivars[0]
        #io.hplot(enmap.enmap(apod,ewcs))
        #sys.exit()
        
        esplits = get_splits(qid,extracter)
        c,civar = coadd(esplits,eivars)
        # io.hplot(down(c*apod))
        ta.update_output("processed",c*civar,inserter)
        ta.update_output("processed_ivar",civar,inserter)
        # kspace.process
        aids.append(qid)
    print(comm.rank, ": Tile %d has arrays " % i, aids) 
    #ilc.build_empirical_cov
    #pmap = ilc.do_ilc
    #ta.update_output("processed",pmap,inserter)
print("Rank %d done" % comm.rank)
pmap = ta.get_final_output("processed")
wmap = ta.get_final_output("processed_ivar")
outmap = pmap/wmap
if comm.rank==0:
    io.hplot(down(outmap),"outmap")
    io.hplot(down(wmap),"wmap")
    mask = sints.get_act_mr3_crosslinked_mask("deep56")
    io.hplot(down(enmap.extract(outmap,mask.shape,mask.wcs)*mask),"moutmap")
    # io.hplot(down(enmap.extract(wmap,mask.shape,mask.wcs)*mask),"mwmap")
    




