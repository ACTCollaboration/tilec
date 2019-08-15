from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from tilec import covtools,ilc
from scipy.optimize import curve_fit


class ASim(object):
    def __init__(self,beams,noises,lknees,alphas,lmins,lmaxs,deg = 20.,px = 2.0):
        theory = cosmology.default_theory()
        shape,wcs = maps.rect_geometry(width_deg=deg,px_res_arcmin=px)
        modlmap = enmap.modlmap(shape,wcs)
        ells = np.arange(modlmap.max())
        cltt = theory.lCl('TT',ells)
        cltt2d = theory.lCl('TT',modlmap)
        self.mgen = maps.MapGen(shape,wcs,cltt[None,None])
        self.ngens = []
        self.n1ds = []
        self.kbeams = []
        narrays = len(beams)
        Ny,Nx = shape[-2:]
        self.tcov = enmap.zeros((Ny,Nx,narrays,narrays),wcs)
        for i,(beam,noise,lknee,alpha) in enumerate(zip(beams,noises,lknees,alphas)):
            ncov = covtools.rednoise(modlmap,noise,lknee=lknee,alpha=alpha)
            n1d = cltt * maps.gauss_beam(beam,ells)**2. + covtools.rednoise(ells,noise,lknee=lknee,alpha=alpha)
            ncov[modlmap<20] = 0
            n1d[ells<20] = 0
            self.ngens.append( maps.MapGen(shape,wcs,ncov[None,None]) )
            self.n1ds.append(n1d.copy())
            kbeam = maps.gauss_beam(beam,modlmap)
            self.kbeams.append(kbeam.copy())
            self.tcov[...,i,i] = cltt2d*kbeam**2. + ncov
        for i in range(narrays):
            for j in range(i+1,narrays):
                self.tcov[...,i,j] = cltt2d*maps.gauss_beam(beams[i],modlmap)*maps.gauss_beam(beams[j],modlmap)
                self.tcov[...,j,i] = self.tcov[...,i,j].copy()
        self.beams = beams
        self.modlmap = modlmap
        self.wcs = wcs
        self.cltt = cltt
        self.modlmap = modlmap
        self.ells = ells

        maxval = self.tcov.max() * 1e4
        for i in range(narrays):
            self.tcov[modlmap<lmins[i],i,i] = maxval
            self.tcov[modlmap>lmaxs[i],i,i] = maxval
        self.tcinv = np.linalg.inv(self.tcov)

    def get_map(self,seed):
        cmb = self.mgen.get_map(seed=(1,1,seed))
        omaps = []
        for i,ngen in enumerate(self.ngens):
            kbeam = maps.gauss_beam(self.beams[i],self.modlmap)
            omap = maps.filter_map(cmb,kbeam) + ngen.get_map(seed=(2,i,seed))
            omaps.append(omap.copy())
        omaps = enmap.enmap(np.stack(omaps),self.wcs)
        return cmb,omaps


def ilc_map(cinv,kbeams,kmaps):
    assert kmaps.ndim==3
    narrays = kmaps.shape[0]
    assert narrays==cinv.shape[-2]==cinv.shape[-1]
    rs = kbeams
    return np.einsum("i...,...ij,j...->...",rs,cinv,kmaps) / np.einsum("i...,...ij,j...->...",rs,cinv,rs)


def ft(x): return enmap.fft(x,normalize='phys')
def ift(x): return enmap.ifft(x,normalize='phys')
def pow(x): 
    k = ft(x)
    return psq(k)
def psq(x,y=None):
    if y is None: y = x
    return np.real(x*y.conj())
def bin(x):
    binner = stats.bin2D(modlmap,bin_edges)
    return binner.bin(x)


def get_ecinv(kmaps,lmins,lmaxs):
    narrays = kmaps.shape[0]
    Ny,Nx = kmaps.shape[-2:]
    ecov = np.zeros((Ny,Nx,narrays,narrays))
    
    for i in range(narrays):
        for j in range(i,narrays):
            x = kmaps[i]
            y = kmaps[j]
            ecov[...,i,j] = covtools.signal_average(psq(x,y),bin_width=160,kind=1,dlspace=True,lmin=20)
            ecov[...,j,i] = ecov[...,i,j].copy()

    maxval = ecov.max() * 1e4
    for i in range(narrays):
        ecov[modlmap<lmins[i],i,i] = maxval
        ecov[modlmap>lmaxs[i],i,i] = maxval

    return np.linalg.inv(ecov)


nsims = 10
beams = [30.,10.,7.,1.5]
noises = [200.,100.,40,20]
lknees = [0,0,0,1]
alphas = [1,1,1,-4]
lmins = [20,20,20,500]
lmaxs = [1000,2000,2000,3000]
narrays = len(beams)
asim = ASim(beams,noises,lknees,alphas,lmins,lmaxs)
bin_edges = np.arange(20,3000,40)
modlmap = asim.modlmap

# imap = asim.get_map(1)
# pl = io.Plotter(xyscale='loglog',xlabel='l',ylabel='C')#D',scalefn = lambda x: x**2./2./np.pi)
# for i in range(narrays):
#     cents,p1d = bin(pow(imap[i]))
#     pl.add(cents,p1d)
#     pl.add(asim.ells,asim.n1ds[i],ls='--')
# pl._ax.set_xlim(18,4000)
# pl.done("lfipow.png")

tcinv = asim.tcinv
kbeams = asim.kbeams

s = stats.Stats()

for i in range(nsims):

    cmb,imap = asim.get_map(i)
    kmaps = ft(imap)
    tilc = ilc_map(tcinv,kbeams,kmaps)
    ecinv = get_ecinv(kmaps,lmins,lmaxs)
    eilc = ilc_map(ecinv,kbeams,kmaps)
    kcmb = ft(cmb)
    cents,tilc1d = bin(psq(tilc,kcmb))
    cents,eilc1d = bin(psq(eilc,kcmb))
    cents,cmb1d = bin(psq(kcmb,kcmb))

    s.add_to_stats('ii',cmb1d)
    s.add_to_stats('ti',tilc1d)
    s.add_to_stats('ei',eilc1d)
    print(i)

s.get_stats()

ii = s.stats['ii']['mean']
ti = s.stats['ti']['mean']
ei = s.stats['ei']['mean']


    
pl = io.Plotter(xyscale='loglog',xlabel='l',ylabel='D',scalefn = lambda x: x**2./2./np.pi)
pl.add(cents,ii)
pl.add(cents,ti)
pl.add(cents,ei)
pl.done("lfirecon.png")

pl = io.Plotter(xyscale='linlin',xlabel='l',ylabel='diff')
pl.add(cents,(ti-ii)/ii,label='true')
pl.add(cents,(ei-ii)/ii,label='emp')
pl.hline(y=0)
pl.done("lfirecondiff.png")
