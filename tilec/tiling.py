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

class TiledAnalysis(object):
    def __init__(self,shape,wcs,pix_width=480,pix_pad=480):
        iNy,iNx = shape[-2:]
        self.numy = iNy // pix_width
        self.numx = iNx // pix_width
        Ny = self.numy * pix_width
        Nx = self.numx * pix_width
        dny = (iNy - Ny)//2
        dnx = (iNx - Nx)//2
        
        self.fpixbox = [[dny,dnx],[Ny+dny,Nx+dnx]]
        self.pboxes = []
        sy = 0
        for i in range(self.numy):
            sx = 0
            for j in range(self.numx):
                self.pboxes.append( [[sy-pix_pad//2,sx-pix_pad//2],[sy+pix_width+pix_pad//2,sx+pix_width+pix_pad//2]] )
                sx += pix_width
            sy += pix_width
            
        
dm = sints.ACTmr3(pickupsub=False)
#imap = enmap.pad(dm.get_coadd("s13","deep6","pa1_f150",srcfree=True,ncomp=None)[0],100)# *0+1
imap = enmap.pad(dm.get_coadd("s14","deep56","pa1_f150",srcfree=True,ncomp=None)[0],300)
io.hplot(enmap.downgrade(imap,8))

ta = TiledAnalysis(imap.shape,imap.wcs)
smap = enmap.extract_pixbox(imap,ta.fpixbox)
io.hplot(enmap.downgrade(smap,8))
for i in range(len(ta.pboxes)):
    pbox = ta.pboxes[i]
    # omap = smap.copy()
    # enmap.insert_at(omap, pbox, np.zeros((480+480,480+480)))
    omap = enmap.extract_pixbox(smap,pbox)
    io.hplot(enmap.downgrade(omap,4))

def get_pixboxes(shape,wcs,width_deg=4.,pad_deg=4.):
    Ny,Nx = shape[-2:]
    ey,ex = enmap.extent(shape,wcs)
    width = np.deg2rad(width_deg)
    pad = np.deg2rad(pad_deg)
    sny = int(width/ey*Ny)
    snx = int(width/ex*Nx)
    pady = int(pad/ey*Ny/2)
    padx = int(pad/ex*Nx/2)
    numy = int(Ny*1./sny)
    numx = int(Nx*1./snx)
    pixboxes = np.zeros((numy,numx,2,2))
    for i in range(numy):
        for j in range(numx):
            sy = i*sny
            ey = (i+1)*sny
            sx = j*snx
            ex = (j+1)*snx

            pbox = np.array([[sy-pady,sx-padx],
                             [ey+pady,ex+padx]])
            pixboxes[i,j] = pbox.copy()

            
    return pixboxes

    
def npix(pbox):
    return pbox[1,0]-pbox[0,0],pbox[1,1]-pbox[0,1]




def linear_crossfade(shape,wcs,deg):
    init = enmap.ones(shape[-2:],wcs)
    Ny,Nx = shape[-2:]
    npix = int(deg*60./px)
    assert Ny%2==0
    assert Nx%2==0
    assert Ny==Nx
    fys = np.ones((Ny,))
    fxs = np.ones((Nx,))
    fys[:npix] = np.linspace(0.,1.,npix)
    fys[Ny-npix:] = np.linspace(0.,1.,npix)[::-1]
    fxs[:npix] = np.linspace(0.,1.,npix)
    fxs[Nx-npix:] = np.linspace(0.,1.,npix)[::-1]
    return fys[:,None] * fxs[None,:]
    

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
