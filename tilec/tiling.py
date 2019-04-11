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


class TiledAnalysis(object):
    def __init__(self,shape,wcs,comm=None,pix_width=480,pix_pad=480,pix_apod=120,pix_cross=240):
        iNy,iNx = shape[-2:]
        self.numy = iNy // pix_width
        self.numx = iNx // pix_width
        Ny = self.numy * pix_width
        Nx = self.numx * pix_width
        dny = (iNy - Ny)//2
        dnx = (iNx - Nx)//2
        self.fpixbox = [[dny,dnx],[Ny+dny,Nx+dnx]]
        self.pboxes = []
        self.ipboxes = []
        sy = 0
        for i in range(self.numy):
            sx = 0
            for j in range(self.numx):
                self.pboxes.append( [[sy-pix_pad//2,sx-pix_pad//2],[sy+pix_width+pix_pad//2,sx+pix_width+pix_pad//2]] )
                self.ipboxes.append( [[sy-pix_pad//2+pix_apod,sx-pix_pad//2+pix_apod],
                                      [sy+pix_width+pix_pad//2-pix_apod,sx+pix_width+pix_pad//2-pix_apod]] )
                sx += pix_width
            sy += pix_width
        if comm is None:
            from orphics import mpi
            comm = mpi.MPI.COMM_WORLD
        self.comm = comm
        N = pix_width + pix_pad
        self.apod = enmap.apod(np.ones((N,N)), pix_apod, profile="cos", fill="zero")
        self.pix_apod = pix_apod
        self.N = N
        self.cN = self.N-self.pix_apod*2
        self.crossfade = self._linear_crossfade(pix_cross)

    def _prepare(self,imap):
        return imap*self.apod

    def _finalize(self,imap):
        return maps.crop_center(imap,self.cN)*self.crossfade

    def tiles(self):
        comm = self.comm
        for i in range(comm.rank, len(self.pboxes), comm.size):
            extracter = lambda x: self._prepare(enmap.extract_pixbox(x,self.pboxes[i]))
            inserter = lambda inp,out: enmap.insert_at(out,self.ipboxes[i],self._finalize(inp),op=np.ndarray.__iadd__)
            yield extracter,inserter
            
    def _linear_crossfade(self,npix):
        init = np.ones((self.cN,self.cN))
        cN = self.cN
        fys = np.ones((cN,))
        fxs = np.ones((cN,))
        fys[:npix] = np.linspace(0.,1.,npix)
        fys[cN-npix:] = np.linspace(0.,1.,npix)[::-1]
        fxs[:npix] = np.linspace(0.,1.,npix)
        fxs[cN-npix:] = np.linspace(0.,1.,npix)[::-1]
        return fys[:,None] * fxs[None,:]


def filter_map(imap):
    modlmap = imap.modlmap()
    ells = np.arange(0,8000,1)
    fcurve = np.exp(-(ells-4000)**2./2./200**2.)
    return maps.filter_map(imap,maps.interp(ells,fcurve)(modlmap))

mask = sints.get_act_mr3_crosslinked_mask("deep56")
dm = sints.ACTmr3(region=mask,pickupsub=False)
imap = enmap.pad(dm.get_coadd("s14","deep56","pa1_f150",srcfree=True,ncomp=None)[0],300)
shape,wcs = imap.shape,imap.wcs


ta = TiledAnalysis(shape,wcs,comm)
omap = enmap.extract_pixbox(enmap.zeros(shape,wcs),ta.fpixbox)
nmap = enmap.extract_pixbox(enmap.zeros(shape,wcs),ta.fpixbox)
cmap = enmap.extract_pixbox(imap,ta.fpixbox)

for ext,ins in ta.tiles():
    emap = ext(cmap)
    emap = filter_map(emap)
    ins(emap,omap)
    ins(emap*0+1,nmap)
    
if comm.rank==0:
    fcmap = filter_map(cmap)
    # io.hplot(enmap.downgrade(omap/nmap,8))
    # io.hplot(enmap.downgrade(fcmap,8))
    io.plot_img(enmap.downgrade(omap/nmap-fcmap,8),lim=1e-1)

    # io.hplot(enmap.downgrade(omap,8))
    # io.plot_img(enmap.downgrade(nmap,8))
    

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
