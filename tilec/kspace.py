from orphics import maps,io,cosmology,stats
from pixell import enmap,curvedsky
import healpy as hp
from actsims import inpaint as inpainting
import numpy as np
import os,sys
from soapack import interfaces as sints
from tilec import utils as tutils


def process(dm,patch,array_id,mask,skip_splits=False,splits_fname=None,inpaint=True,fn_beam=None,cache_inpaint_geometries=True,verbose=True,plot_inpaint_path=None):
    """
    Return (nsplits,Ny,Nx) fourier transform
    Return (Ny,Nx) fourier transform of coadd

    This function applies no corrections for masks.
    """
    qid = array_id
    if dm.name=='act_mr3':
        season,array1,array2 = sints.arrays(qid,'season'),sints.arrays(qid,'array'),sints.arrays(qid,'freq')
        array = '_'.join([array1,array2])
        pixwin = True
    elif dm.name=='planck_hybrid':
        season,patch,array = None,None,sints.arrays(qid,'freq')
        pixwin = False
    wins = dm.get_splits_ivar(season=season,patch=patch,arrays=[array],ncomp=None)[0,:,0,:,:]
    if splits_fname is None: 
        splits = dm.get_splits(season=season,patch=patch,arrays=[array],ncomp=3,srcfree=True)[0,:,:,:,:]
    else:
        splits = enmap.read_map(splits_fname)
    assert splits.ndim==4
    nsplits = splits.shape[0]
    assert nsplits==2 or nsplits==4
    # Inpaint
    if inpaint and dm.name=='act_mr3':
        rsplits = []
        for i in range(nsplits): 
            result = inpainting.inpaint_map_white(splits[i],wins[i],fn_beam,
                                                  union_sources_version=None,plots=False,
                                                  cache_name="qid_%s_splitnum_%d" % (qid,i) if cache_inpaint_geometries else None,
                                                  verbose=verbose)
            rsplits.append(result[0,:,:].copy())
            if plot_inpaint_path is not None:
                io.hplot(splits[i][0,:,:],"%s/uninpainted_qid_%s_splitnum_%d" % (plot_inpaint_path,qid,i),grid=True,color='planck')
                io.hplot(result[0,:,:],"%s/inpainted_qid_%s_splitnum_%d" % (plot_inpaint_path,qid,i),grid=True,color='planck')
        rsplits = enmap.enmap(np.stack(rsplits),splits.wcs)
    else:
        rsplits = splits[:,0,:,:]
    kdiffs,kcoadd = process_splits(qid,rsplits,wins,mask,skip_splits=skip_splits,do_fft_splits=False,pixwin=pixwin)
    return kdiffs,kcoadd,wins

def process_splits(qid,splits,wins,mask,skip_splits=False,do_fft_splits=False,pixwin=False,hybrid_beam=False):
    assert wins.ndim>2
    with np.errstate(divide='ignore', invalid='ignore'):
        coadd = (splits*wins).sum(axis=0)/wins.sum(axis=0)
    coadd[~np.isfinite(coadd)] = 0
    Ny,Nx = splits.shape[-2:]
    assert coadd.shape == (Ny,Nx)
    if hybrid_beam and (qid in ['p01','p02']):
        raise NotImplementedError
        # If this option is on, we do the following
        # 1. get alms of coadd
        # 2. get beam+planck-pixwin in ell space
        # 3. divide it out
        # 4. ISHT the map
        # 5. multiply in Fourier space by the same beam_planck-pixwin
        wcs = splits.wcs
        modlmap = splits.modlmap()
        lmax = int(modlmap.max())+1
        imap = (coadd*mask)[None]
        alms = curvedsky.map2alm(imap,spin=0,lmax=lmax)[0]
        ells = np.arange(0,lmax)
        def bfunc(x): 
            rbeam = tutils.get_kbeam(qid,x,sanitize=True,planck_pixwin=True)
            rbeam[~np.isfinite(rbeam)] = 0
            return rbeam
        ibeam = 1./bfunc(ells)
        ibeam[~np.isfinite(ibeam)] = 0
        oalms = hp.almxfl(alms,ibeam)
        omap = enmap.zeros((Ny,Nx),wcs)
        omap = curvedsky.alm2map(oalms,omap)
        kcoadd = enmap.enmap(enmap.fft(omap,normalize='phys') * bfunc(modlmap),wcs)
    else:
        kcoadd = enmap.enmap(enmap.fft(coadd*mask,normalize='phys'),wins.wcs)




    if pixwin: pwin = tutils.get_pixwin(coadd.shape[-2:])
    else: pwin = 1
    kcoadd = kcoadd / pwin
    if not(skip_splits):
        data = (splits-coadd)*mask # !!!! wins removed, was (splits-coadd)*wins*mask earlier
        kdiffs = enmap.fft(data,normalize='phys')
        kdiffs = kdiffs / pwin
    else:
        ksplits = None
    if do_fft_splits:
        ksplits = enmap.fft(splits*wins*mask,normalize='phys')
        ksplits = ksplits / pwin
        return kdiffs,kcoadd,ksplits
    else:
        return kdiffs,kcoadd


