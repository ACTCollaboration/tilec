from __future__ import print_function
from tilec import needlets,fg
from orphics import io,maps
from pixell import enmap,utils
import numpy as np
import os,sys

dfact = 4
# #area = 500.. * utils.degree**2
# beam = 1.5 * utils.arcmin
# beam_area = 2* np.pi * (1-np.cos(beam))
# #Nmodes = area / beam_area
# #(Nmodes/area) * 2* np.pi * (1-np.cos(sm_theta)) = 4225
# cos = 1 - (4225 * beam_area / 2 /np.pi)
# theta = np.arccos(cos)
# print(theta / utils.degree)
# sys.exit()

lmax = 30000
fwhms = np.array([600., 300., 120., 60., 30., 15., 10., 7.5, 5.,4.,3.,2.,1.0])
nspect = fwhms.size + 1
filters = needlets.gaussian_needlets(lmax,fwhms)

# ls = np.arange(filters.shape[1])
# pl = io.Plotter(xyscale='loglin',xlabel='l',ylabel='f')
# for i in range(filters.shape[0]): pl.add(ls[2:],filters[i,2:],label=str(i))
# pl.add(ls[2:],(filters[:,2:]**2.).sum(axis=0),color='k')
# pl.legend(loc='center left',bbox_to_anchor=(1,0.5))
# pl.done('filters.png')


bshape,bwcs = enmap.read_map_geometry('/home/msyriac/data/act/maps/mr3/s13_deep5_pa1_f150_nohwp_night_3pass_4way_coadd_map_srcfree.fits')
m2 = enmap.read_map('/home/msyriac/data/act/maps/mr3/s14_deep56_pa1_f150_nohwp_night_3pass_4way_coadd_map_srcfree.fits',sel=np.s_[0,...])
pbox = enmap.pixbox_of(bwcs,m2.shape,m2.wcs)
m1 = enmap.read_map('/home/msyriac/data/act/maps/mr3/s13_deep5_pa1_f150_nohwp_night_3pass_4way_coadd_map_srcfree.fits',pixbox=pbox,sel=np.s_[0,...])
mask1 = enmap.extract(enmap.read_map('/home/msyriac/data/act/maps/steve/padded_v1/deep5.fits'),m1.shape,m1.wcs)
mask2 = enmap.extract(enmap.read_map('/home/msyriac/data/act/maps/steve/padded_v1/deep56.fits'),m2.shape,m2.wcs)

bshape,bwcs = enmap.read_map_geometry('/home/msyriac/data/planck/sigurd/archived/planck_hybrid_143_2way_0_map_I_srcfree.fits')
pbox = enmap.pixbox_of(bwcs,m2.shape,m2.wcs)
m3 = enmap.read_map('/home/msyriac/data/planck/sigurd/archived/planck_hybrid_143_2way_0_map_I_srcfree.fits',pixbox=pbox)
m1 = enmap.downgrade(m1,dfact)
m2 = enmap.downgrade(m2,dfact)
m3 = enmap.downgrade(m3,dfact)

mask1 = enmap.downgrade(mask1,dfact)
mask2 = enmap.downgrade(mask2,dfact)
mask3,_ = maps.get_taper_deg(m3.shape,m3.wcs,2.)

modlmap = mask1.modlmap()

# io.plot_img(m1,lim=300)
# io.plot_img(m2,lim=300)
# io.plot_img(m3,lim=300)
# io.plot_img(mask1)
# io.plot_img(mask2)
# io.plot_img(mask3)
# io.plot_img(m1*mask1,lim=300)
# io.plot_img(m2*mask2,lim=300)
# io.plot_img(m3*mask3,lim=300)

filt = maps.gauss_beam(modlmap,1.5)/maps.gauss_beam(modlmap,7.0)
filt[modlmap>2000] = 0
m3 = maps.filter_map(m3*mask3,filt)
# m3 = m3 * mask3

# masks = [mask3]
# nmaps = np.asarray(masks).shape[0]
# fmaps = np.zeros((nspect,nmaps,*mask1.shape))
# imaps = [m3]

# masks = [mask2,mask3]
# nmaps = np.asarray(masks).shape[0]
# fmaps = np.zeros((nspect,nmaps,*mask1.shape))
# imaps = [m2*mask2,m3]


masks = [mask1,mask2,mask3]
nmaps = np.asarray(masks).shape[0]
fmaps = np.zeros((nspect,nmaps,*mask1.shape))
imaps = [m1*mask1,m2*mask2,m3]

# masks = [mask1,mask2]
# nmaps = np.asarray(masks).shape[0]
# fmaps = np.zeros((nspect,nmaps,*mask1.shape))
# imaps = [m1*mask1,m2*mask2]


print(fmaps.shape)

# rmaps = np.zeros((nmaps,*mask1.shape))
# for i in range(nmaps):
#     io.plot_img(imaps[i],lim=300)
    
for i in range(nspect):
    for j in range(nmaps):
        fmap = maps.filter_map(imaps[j],maps.interp(ls,filters[i])(modlmap))
        # rmaps[j] = rmaps[j] + maps.filter_map(fmap.copy(),maps.interp(ls,filters[i])(modlmap))
        print(fmap.shape)
        fmaps[i,j] = fmap.copy()
        # io.plot_img(fmaps[i,j],lim=300)
fmaps = enmap.enmap(fmaps,mask1.wcs)

# for i in range(nmaps):
#     io.plot_img(rmaps[i],lim=300)

cinvs = np.zeros((nspect,*mask1.shape,nmaps,nmaps))

for i in range(nspect):
    covs = np.zeros((*mask1.shape,nmaps,nmaps))
    for j in range(nmaps):
        for k in range(j,nmaps):
            mask1 = masks[j]
            mask2 = masks[k]
            # w2 = np.mean(mask1*mask2)
            cov = maps.filter_map(fmaps[i,j]*fmaps[i,k],maps.gauss_beam(modlmap,2*97.)) #/ w2
            # cov = fmaps[i,j]*fmaps[i,k] #/ w2
            covs[...,j,k] = cov.copy()
            covs[...,k,j] = cov.copy()
            if j==k: covs[mask1<0.99,j,j] = 1e8 #np.inf
            # if j==k: io.plot_img(cov)
            
    cinvs[i,...] = np.linalg.inv(covs[i,...])
    print(i)

for i in range(nspect):
    for j in range(nmaps):
        mask1 = masks[j]
        cinvs[i,...,j,j][mask1<0.99] = 0


omap = 0
r = np.asarray([fg.get_mix(143, 'tSZ')[0]] * nmaps)
for i in range(nspect):
    cinv = cinvs[i]
    idenom = 1./np.einsum('ijl,l->ij',np.einsum('k,ijkl->ijl',r,cinv),r)
    idenom[~np.isfinite(idenom)] = 0
    omap += maps.filter_map(enmap.enmap(np.einsum('k,kij->ij' ,r,np.einsum('ijkl,lij->kij',cinv,fmaps[i])) * idenom,mask1.wcs),maps.interp(ls,filters[i])(modlmap))

#io.plot_img(omap)
# io.plot_img(omap,'ilc.png',lim=300)
# io.plot_img(m3,'planck.png',lim=300)

io.hplot(omap,'ilc',range=300)
io.hplot(m3,'planck',range=300)

io.hplot(omap,'gilc',color='gray')
io.hplot(m3,'gplanck',color='gray')
