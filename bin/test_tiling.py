"""

We will simulate a part-sky lensed Q/U map at 2 arcmin resolution.

In order to make a tiling lensing reconstruction, one has to:
1. divide the sky into overlapping tiles
2. extract each tile and filter it (e.g. constant covariance wiener filter based on ILC empirical covmat)
3. calculate the normalization for that particular filter
4. reconstruct QE
5. apply the normalization
6. simulate the above and subtract a meanfield
7. insert the reconstruction back into the full map with a cross-fading window

"""


from __future__ import print_function
from orphics import maps,io,cosmology,stats,lensing
from pixell import enmap,lensing as enlensing,powspec,utils
import numpy as np
import os,sys
from enlib import bench
from soapack import interfaces as sints
from orphics import io,maps
from orphics import mpi
comm = mpi.MPI.COMM_WORLD
from tilec import tiling


np.random.seed(1)


def filter_map(imap):
    modlmap = imap.modlmap()
    ells = np.arange(0,8000,1)
    fcurve = np.exp(-(ells-4000)**2./2./200**2.)
    return maps.filter_map(imap,maps.interp(ells,fcurve)(modlmap))

#mask = sints.get_act_mr3_crosslinked_mask("deep56")
#dm = sints.ACTmr3(region=mask,pickupsub=False)
# dm = sints.ACTmr3(pickupsub=False)
# imap = enmap.pad(dm.get_coadd("s14","deep56","pa1_f150",srcfree=True,ncomp=None)[0],300)
# shape,wcs = imap.shape,imap.wcs

# shape,wcs = maps.rect_geometry(width_deg=20.,px_res_arcmin=0.5)
# imap = enmap.rand_map(shape,wcs,np.ones((1,1,shape[0],shape[1])))

oshape,owcs = enmap.fullsky_geometry(res=np.deg2rad(0.5/60.))
Ny = oshape[-2:][0]
deg = 5
npix = deg/(0.5/60.)
shape,wcs = enmap.slice_geometry(oshape,owcs,np.s_[int(Ny//2-npix):int(Ny//2+npix),:])
imap = enmap.rand_map(shape,wcs,np.ones((1,1,shape[0],shape[1])))


ta = tiling.TiledAnalysis(shape,wcs,comm)
ta.initialize_output("out")

for ext,ins in ta.tiles():
    emap = ext(imap)
    emap = filter_map(emap)
    ta.update_output("out",emap,ins)
    
outmap = ta.get_final_output("out")
# print(comm.rank)
# io.plot_img(outmap,"rank_%d" % comm.rank)
if comm.rank==0:
    fcmap = filter_map(imap)
    io.hplot(enmap.downgrade(imap,8))
    io.hplot(enmap.downgrade(outmap,8))
    io.hplot(enmap.downgrade(fcmap,8))
    io.plot_img(enmap.downgrade(outmap-fcmap,8),lim=1)

# brmap = enmap.zeros(observed.shape[-2:],observed.wcs)
# bwrmap = enmap.zeros(observed.shape[-2:],observed.wcs)
# shape,wcs = observed.shape,observed.wcs
# epboxes = get_pixboxes(shape,wcs,width_deg,pad_deg)
# pboxes = get_pixboxes(shape,wcs,width_deg,pad_deg-2.*rtap_deg)
# for i in range(pboxes.shape[0]):
#     for j in range(pboxes.shape[1]):
#         omap = observed.copy()
#         #print(npix(pboxes[i,j]))
#         emap = enmap.extract_pixbox(omap,epboxes[i,j],wrap=shape[-2:])
#         print("Min ell: ", maps.minimum_ell(emap.shape,emap.wcs))


#         taper,w2 = maps.get_taper_deg(emap.shape,emap.wcs,taper_width_degrees = rtap_deg,pad_width_degrees = 0.)
        
#         # io.hplot(emap*taper)
#         rmap,_ = reconstruct(emap*taper,beam_arcmin,noise_uk_arcmin,noise_uk_arcmin*np.sqrt(2.))

#         cropped = maps.crop_center(rmap,int((width_deg+pad_deg-2*rtap_deg)*60./px))
#         wtaper = linear_crossfade(cropped.shape,cropped.wcs,cross_deg)
        
#         enmap.insert_at(brmap,pboxes[i,j],cropped*wtaper,wrap=shape[-2:],op=np.ndarray.__iadd__)
#         enmap.insert_at(bwrmap,pboxes[i,j],wtaper,wrap=shape[-2:],op=np.ndarray.__iadd__)
#         print(emap.shape)
#         # io.hplot(omap[0])
#         #io.hplot(emap)

# io.hplot(brmap,"trecon")
# io.hplot(bwrmap,"twmap")
# io.plot_img(bwrmap,"twmap_mat.png")
# brmap = brmap/bwrmap
# io.hplot(brmap,"brmap")

# theory = cosmology.default_theory()
# shape,wcs = observed.shape,observed.wcs
# modlmap = enmap.modlmap(shape,wcs)

# rmap,qest = reconstruct(observed,beam_arcmin,noise_uk_arcmin,noise_uk_arcmin*np.sqrt(2.))
# kellmin = 20
# kellmax = 3000

# kk2d = theory.gCl('kk',modlmap)
# nkk2d = qest.N.Nlkk[polcomb]
# wiener2d = np.nan_to_num(kk2d/(kk2d+nkk2d))
# wiener2d[modlmap<kellmin] = 0
# wiener2d[modlmap>kellmax] = 0

# bkappaf = maps.filter_map(rmap,wiener2d)
# tkappaf = maps.filter_map(brmap,wiener2d)
# bphif = lensing.kappa_to_phi(bkappaf,modlmap)
# tphif = lensing.kappa_to_phi(tkappaf,modlmap)
# phii = lensing.kappa_to_phi(maps.filter_map(kappa,wiener2d),modlmap)
# io.hplot(bphif,"bphif")
# io.hplot(tphif,"tphif")
# io.hplot(phii,"phii")

# bin_edges = np.geomspace(kellmin,kellmax,30)
# cents,ii = maps.binned_power(kappa,bin_edges)
# cents,bi = maps.binned_power(kappa,bin_edges,imap2=rmap)
# cents,ti = maps.binned_power(kappa,bin_edges,imap2=brmap)
# cents,bb = maps.binned_power(rmap,bin_edges)
# cents,tt = maps.binned_power(brmap,bin_edges)

# pl = io.Plotter(xyscale='loglog',xlabel='$L$',ylabel='$C_L$')
# pl.add(cents,ii,color='k')
# pl.add(cents,bi,color='C0',label='untiled')
# pl.add(cents,ti,color='C1',label='tiled')
# pl.add(cents,bb,color='C0',ls="--")
# pl.add(cents,tt,color='C1',ls="--")
# pl.done("clkk.png")
