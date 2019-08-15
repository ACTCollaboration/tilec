from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from tilec import covtools,ilc
from scipy.optimize import curve_fit

deg = 20.
px = 2.0

theory = cosmology.default_theory()
shape,wcs = maps.rect_geometry(width_deg=deg,px_res_arcmin=px)
modlmap = enmap.modlmap(shape,wcs)
ells = np.arange(modlmap.max())
cltt = theory.lCl('TT',ells)


mgen = maps.MapGen(shape,wcs,cltt[None,None])
noise = [10,20]
ngen1 = maps.MapGen(shape,wcs,(ells*0 + (noise[0]*np.pi/180./60.)**2.)[None,None])
ngen2 = maps.MapGen(shape,wcs,(ells*0 + (noise[1]*np.pi/180./60.)**2.)[None,None])

cov = enmap.enmap(np.zeros((shape[0],shape[1],2,2)),wcs)
for i in range(2):
    for j in range(2):
        cov[...,i,j] = maps.interp(ells,cltt)(modlmap) + int(i==j) * (noise[i]*np.pi/180./60.)**2.

cinv = np.linalg.inv(cov)
nsims = 30

np.random.seed(1)

bin_edges = np.arange(80,3000,40)
binner = stats.bin2D(modlmap,bin_edges)

s = stats.Stats()


gellmax = modlmap.max()
ells = np.arange(0,gellmax,1)
ctheory = ilc.CTheory(ells)
slmin = 80
minell = maps.minimum_ell(shape,wcs)
fitmax = 600
fitmin = slmin
dell = 2*minell
fbin_edges = np.arange(fitmin,fitmax,dell)
fbinner = stats.bin2D(modlmap,fbin_edges)
fcents = fbinner.centers

for i in range(nsims):
    print(i)
    cmb = mgen.get_map(seed=(1,i))
    n1 = ngen1.get_map(seed=(2,i))
    n2 = ngen2.get_map(seed=(3,i))
    

    kmap0 = enmap.fft(cmb,normalize='phys')
    kmap1 = enmap.fft(cmb+n1,normalize='phys')
    kmap2 = enmap.fft(cmb+n2,normalize='phys')

    kmaps = [kmap1,kmap2]
    icov = np.zeros((shape[0],shape[1],2,2))
    ncov = np.zeros((shape[0],shape[1],2,2))
    lmin = 80
    lmax = 7000
    for p in range(2):
        for q in range(2):
            power = np.real(kmaps[p]*kmaps[q].conj())
            icov[...,p,q] = covtools.signal_average(power,bin_width=80,kind=3,dlspace=True,lmin=lmin)
            #icov[...,p,q] = covtools.signal_average(enmap.enmap(cov[...,p,q],wcs),bin_width=80,kind=3,dlspace=True,lmin=lmin)
            
            ncov[...,p,q] = icov[...,p,q].copy()

            np.random.seed((4,i,p,q))
            stoch = (1+np.random.normal(scale=0.01))
            print(100-stoch*100.)
            ncov[...,p,q][modlmap<600] = icov[...,p,q][modlmap<600].copy() * stoch
            #ncov[modlmap<600,p,q] = cov[modlmap<600,p,q].copy()

            # f1 = 150 ; f2 = 150
            # ffunc = lambda d,x: fbinner.bin(maps.interp(ells,ctheory.get_theory_cls(f1,f2,a_cmb=x))(modlmap))[1]
            # res,_ = curve_fit(ffunc,fcents,fbinner.bin(power)[1],p0=[1],bounds=([0.2],[1.8]))                
            # fcmb = res
            # print(fcmb)
            # cfit = maps.interp(ells,ctheory.get_theory_cls(f1,f2,a_cmb=fcmb))(modlmap)
            # ncov[modlmap<600,p,q] = cfit[modlmap<600].copy()

            if p==q: 
                icov[modlmap<=lmin,p,q] = cov.max()*10000
                icov[modlmap>=lmax,p,q] = cov.max()*10000
                ncov[modlmap<=lmin,p,q] = cov.max()*10000
                ncov[modlmap>=lmax,p,q] = cov.max()*10000
            #io.power_crop(icov[...,p,q],200,"dscov_%d_%d.png" % (p,q))
            #icov[...,p,q] = cov[...,p,q]

    icinv = np.linalg.inv(icov)
    ncinv = np.linalg.inv(ncov)

    
    
    ks = np.stack([kmap1,kmap2])
    rs = np.ones((2,))
    kcoadd = np.einsum("i,...ij,j...->...",rs,cinv,ks) / np.einsum("i,...ij,j->...",rs,cinv,rs)
    ikcoadd = np.einsum("i,...ij,j...->...",rs,icinv,ks) / np.einsum("i,...ij,j->...",rs,icinv,rs)
    nkcoadd = np.einsum("i,...ij,j...->...",rs,ncinv,ks) / np.einsum("i,...ij,j->...",rs,ncinv,rs)
    
    p2d = np.real(kcoadd*kmap0.conj())
    cents,p1d = binner.bin(p2d)
    s.add_to_stats("p1d",p1d)

    p2d = np.real(kmap0*kmap0.conj())
    cents,p1d0 = binner.bin(p2d)
    s.add_to_stats("p1d0",p1d0)

    p2d = np.real(ikcoadd*kmap0.conj())
    cents,p1d = binner.bin(p2d)
    s.add_to_stats("ip1d",p1d)

    p2d = np.real(nkcoadd*kmap0.conj())
    cents,p1d = binner.bin(p2d)
    s.add_to_stats("np1d",p1d)



s.get_stats()

p1d = s.stats['p1d']['mean']
p1d0 = s.stats['p1d0']['mean']
ip1d = s.stats['ip1d']['mean']
np1d = s.stats['np1d']['mean']

pl = io.Plotter(xyscale='loglog',scalefn = lambda x: x**2./2./np.pi,xlabel='l',ylabel='D')
pl.add(ells,cltt)
pl.add(cents,p1d)
pl.done("simpleilc.png")

pl = io.Plotter(xyscale='linlin',xlabel='l',ylabel='D')
pl.add(cents,(p1d-p1d0)/p1d0)
pl.add(cents,(ip1d-p1d0)/p1d0,ls="-")
pl.add(cents,(np1d-p1d0)/p1d0,ls="--")
pl._ax.set_xlim(70,1000)
pl._ax.set_ylim(-0.02,0.02)
pl.hline()
pl.done("dsimpleilc.png")

