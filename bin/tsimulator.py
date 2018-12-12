from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from orphics import maps,io,cosmology,lensing,stats,mpi
from pixell import enmap,lensing as enlensing,utils
import numpy as np
import os,sys
from szar import foregrounds as fg
from tilec import utils as tutils,covtools

beams,freqs,noises,lknees,alphas,nsplits = np.loadtxt("input/simple_sim.txt",unpack=True)

nsims = 1
comm,rank,my_tasks = mpi.distribute(nsims)

def process(kmaps,ellmax=None):
    ellmax = lmax if ellmax is None else ellmax
    kout = enmap.zeros((Ny,Nx),wcs,dtype=np.complex128).reshape(-1)
    kout[modlmap.reshape(-1)<lmax] = np.nan_to_num(kmaps.copy())
    kout = enmap.enmap(kout.reshape((Ny,Nx)),wcs)
    return kout

def compute(ik1,ik2,tag):
    pcross2d = fc.f2power(ik1,ik2)
    cents,p1d = binner.bin(pcross2d)
    s.add_to_stats(tag,p1d.copy())

def ncompute(ik,nk,tag):
    pauto2d = fc.f2power(ik,ik) - fc.f2power(nk,nk)
    cents,p1d = binner.bin(pauto2d)
    s.add_to_stats(tag,p1d.copy())
    

class TSimulator(object):
    def __init__(self,shape,wcs,beams,freqs,noises,lknees,alphas,nsplits,pss,nu0,lmax=6000):
        self.nu0 = nu0
        self.freqs = freqs
        theory = cosmology.default_theory()
        ells = np.arange(0,lmax,1)
        cltt = theory.uCl('TT',ells)
        self.cseed = 0
        self.kseed = 1
        self.nseed = 2
        self.fgen = maps.MapGen((pss.shape[0],)+shape[-2:],wcs,pss)
        self.fgen = maps.MapGen((3,)+shape[-2:],wcs,pss)
        self.cgen = maps.MapGen(shape[-2:],wcs,cltt[None,None])
        self.shape, self.wcs = shape,wcs
        self.modlmap = enmap.modlmap(shape,wcs)
        self.arrays = range(len(freqs))
        self.nsplits = nsplits
        self.ngens = []
        self.kbeams = []
        for array in self.arrays:
            ps_noise = cosmology.noise_func(ells,0.,noises[array],lknee=lknees[array],alpha=alphas[array])
            self.ngens.append( maps.MapGen(shape[-2:],wcs,ps_noise[None,None]*nsplits[array]) )
            self.kbeams.append( maps.gauss_beam(self.modlmap,beams[array]) )

    def get_corr(self,seed):
        fmap = self.fgen.get_map(seed=(self.kseed,seed))
        return fmap

    def _lens(self,unlensed,kappa,lens_order=5):
        self.kappa = kappa
        alpha = lensing.alpha_from_kappa(kappa,posmap=enmap.posmap(self.shape,self.wcs))
        lensed = enlensing.displace_map(unlensed, alpha, order=lens_order)
        return lensed

    def get_sim(self,seed):
        kappa,tsz,cib = self.get_corr(seed)
        unlensed = self.cgen.get_map(seed=(self.cseed,seed))
        lensed = self._lens(unlensed,kappa)
        self.lensed = lensed.copy()
        tcmb = 2.726e6
        self.y = tsz.copy()/tcmb/fg.ffunc(self.nu0)
        observed = []
        noises = []
        for array in self.arrays:
            scaled_tsz = tsz * fg.ffunc(self.freqs[array]) / fg.ffunc(self.nu0)
            scaled_cib = cib * fg.cib_nu(self.freqs[array]) / fg.cib_nu(self.nu0)
            sky = lensed #+ scaled_tsz + scaled_cib
            beamed = maps.filter_map(sky,self.kbeams[array])
            observed.append([])
            noises.append([])
            for split in range(self.nsplits[array]):
                noise = self.ngens[array].get_map(seed=(self.nseed,seed,split))
                observed[array].append(beamed+noise)
                noises[array].append(noise)
        return observed,noises


lmax = 4000
shape,wcs = maps.rect_geometry(width_deg=35.,height_deg=15.,px_res_arcmin=1.5)
ells = np.arange(0,lmax,1)
theory = cosmology.default_theory()

nu0 = 150.
cibkcorr = fg.kappa_cib_corrcoeff(ells)
ycorr = fg.y_kappa_corrcoeff(ells)

cltt = theory.lCl('tt',ells)
clkk = theory.gCl('kk',ells)
clss = fg.power_tsz(ells,nu0)
clsc = fg.power_tsz_cib(ells,nu0)
clcc = fg.power_cibc(ells,nu0)
clkc = cibkcorr * np.sqrt(clkk*clcc)
clks = ycorr*np.sqrt(clkk*clss)

# pl = io.Plotter(yscale='log',scalefn=lambda x: x**2)
# pl.add(ells,cltt,label="tt")
# pl.add(ells,clss,label="sz")
# pl.add(ells,-clsc,label="sz x cib")
# pl.add(ells,clcc,label="cib")
# pl.done()

ncomp = 3
ps = np.zeros((ncomp,ncomp,ells.size))
ps[0,0] = clkk
ps[1,1] = clss
ps[2,2] = clcc
ps[1,2] = ps[2,1] = clsc
ps[0,1] = ps[1,0] = clks
ps[0,2] = ps[2,0] = clkc

# fgen = maps.MapGen((3,)+shape[-2:],wcs,ps)
# comps = fgen.get_map(0)
# io.plot_img(comps[0])
# io.plot_img(comps[1])
# io.plot_img(comps[2])

tsim = TSimulator(shape,wcs,beams,freqs,noises,lknees,alphas,nsplits.astype(np.int),ps,nu0)
modlmap = tsim.modlmap
fc = maps.FourierCalc(tsim.shape,tsim.wcs)
narrays = len(tsim.arrays)
iells = modlmap[modlmap<lmax].reshape(-1) # unraveled disk
nells = iells.size
Scov = np.zeros((narrays,narrays,nells))
Ncov = np.zeros((narrays,narrays,nells))
Ny,Nx = shape[-2:]
tcmb = 2.726e6
yresponses = fg.ffunc(tsim.freqs)*tcmb
cresponses = yresponses*0.+1.

bin_edges = np.arange(100,lmax-500,80)
binner = stats.bin2D(modlmap,bin_edges)
cents = binner.centers

s = stats.Stats(comm)

for task in my_tasks:
    i = task
    isim,isimnoise = tsim.get_sim(i)
    coadds = []
    ikmaps = []
    inkmaps = []
    for array in tsim.arrays:
        coadd = sum(isim[array])/tsim.nsplits[array]
        ncoadd = sum(isimnoise[array])/tsim.nsplits[array]
        coadds.append(coadd)
        _,_,kcoadd = fc.power2d(coadds[-1])
        _,_,kncoadd = fc.power2d(ncoadd)
        ikmaps.append(  np.nan_to_num(kcoadd/tsim.kbeams[array]) )
        inkmaps.append(  np.nan_to_num(kncoadd/tsim.kbeams[array]) )
        
    for aindex1 in range(narrays):
        for aindex2 in range(aindex1,narrays) :
            scov,ncov,autos = tutils.ncalc(isim,coadds,aindex1,aindex2,fc,do_coadd_noise=False)
            dscov = covtools.signal_average(scov,bin_width=40)
            dncov,_,_ = covtools.noise_average(ncov,radial_fit=False) if (aindex1==aindex2)  else (0.,None,None)
            dncov = np.nan_to_num(dncov/tsim.kbeams[aindex1]/tsim.kbeams[aindex2])
            dscov = np.nan_to_num(dscov/tsim.kbeams[aindex1]/tsim.kbeams[aindex2])
            Scov[aindex1,aindex2] = dscov[modlmap<lmax].reshape(-1)
            Ncov[aindex1,aindex2] = dncov[modlmap<lmax].reshape(-1)
            if aindex1!=aindex2: Scov[aindex2,aindex1] = Scov[aindex1,aindex2].copy()
            if aindex1!=aindex2: Ncov[aindex2,aindex1] = Ncov[aindex1,aindex2].copy()

    ilensed = tsim.lensed
    _,iklensed,_ = fc.power2d(ilensed)
    iy = tsim.y
    _,iky,_ = fc.power2d(iy)
            
    Cov = Scov + Ncov
    Cov = np.rollaxis(Cov,2)
    icinv = utils.eigpow(Cov,-1)
    Cinv = np.rollaxis(icinv,0,3)
    ikmaps = np.stack(ikmaps)
    ikmaps[:,modlmap>lmax] = 0
    inkmaps = np.stack(inkmaps)
    inkmaps[:,modlmap>lmax] = 0
    kmaps = ikmaps.reshape((narrays,Ny*Nx))[:,modlmap.reshape(-1)<lmax]
    nkmaps = inkmaps.reshape((narrays,Ny*Nx))[:,modlmap.reshape(-1)<lmax]
    iksilc = process(maps.silc(kmaps,Cinv,yresponses))
    inksilc = process(maps.silc(nkmaps,Cinv,yresponses))
    compute(iksilc,iky,"y_silc_cross")
    ncompute(iksilc,inksilc,"y_silc_auto")
    iksilc = process(maps.cilc(kmaps,Cinv,yresponses,cresponses))
    inksilc = process(maps.cilc(nkmaps,Cinv,yresponses,cresponses))
    compute(iksilc,iky,"y_cilc_cross")
    ncompute(iksilc,inksilc,"y_cilc_auto")
    
    iksilc = process(maps.silc(kmaps,Cinv,cresponses))
    inksilc = process(maps.silc(nkmaps,Cinv,cresponses))
    compute(iksilc,iklensed,"cmb_silc_cross")
    ncompute(iksilc,inksilc,"cmb_silc_auto")
    iksilc = process(maps.cilc(kmaps,Cinv,cresponses,yresponses))
    inksilc = process(maps.cilc(nkmaps,Cinv,cresponses,yresponses))
    compute(iksilc,iklensed,"cmb_cilc_cross")
    ncompute(iksilc,inksilc,"cmb_cilc_auto")
    if rank==0: print ("Rank 0 done with task ", task+1, " / " , len(my_tasks))

s.get_stats()

if rank==0:
    cmb_silc_cross = s.stats["cmb_silc_cross"]['mean']
    cmb_cilc_cross = s.stats["cmb_cilc_cross"]['mean']
    y_silc_cross = s.stats["y_silc_cross"]['mean']
    y_cilc_cross = s.stats["y_cilc_cross"]['mean']
    ecmb_silc_cross = s.stats["cmb_silc_cross"]['errmean']
    ecmb_cilc_cross = s.stats["cmb_cilc_cross"]['errmean']
    ey_silc_cross = s.stats["y_silc_cross"]['errmean']
    ey_cilc_cross = s.stats["y_cilc_cross"]['errmean']
    cmb_silc_auto = s.stats["cmb_silc_auto"]['mean']
    cmb_cilc_auto = s.stats["cmb_cilc_auto"]['mean']
    y_silc_auto = s.stats["y_silc_auto"]['mean']
    y_cilc_auto = s.stats["y_cilc_auto"]['mean']
    ecmb_silc_auto = s.stats["cmb_silc_auto"]['errmean']
    ecmb_cilc_auto = s.stats["cmb_cilc_auto"]['errmean']
    ey_silc_auto = s.stats["y_silc_auto"]['errmean']
    ey_cilc_auto = s.stats["y_cilc_auto"]['errmean']
    ells = np.arange(0,lmax,1)
    cltt = theory.lCl('TT',ells)
    clyy = fg.power_y(ells)

    pl = io.Plotter(yscale='log',scalefn=lambda x: x**2,xlabel='l',ylabel='D')
    pl.add(ells,cltt)
    pl.add_err(cents,cmb_silc_cross,yerr=ecmb_silc_cross,marker="o",ls="none",label='standard')
    pl.add_err(cents+10,cmb_cilc_cross,yerr=ecmb_cilc_cross,marker="o",ls="none",label='constrained')
    pl.add_err(cents,cmb_silc_auto,yerr=ecmb_silc_auto,marker="d",ls="none",label='standard')
    pl.add_err(cents+10,cmb_cilc_auto,yerr=ecmb_cilc_auto,marker="d",ls="none",label='constrained')
    pl.done(io.dout_dir+"cmb_cross.png")


    pl = io.Plotter(yscale='log',scalefn=lambda x: x**2,xlabel='l',ylabel='D')
    pl.add(ells,clyy)
    pl.add_err(cents,y_silc_cross,yerr=ey_silc_cross,marker="o",ls="none",label='standard')
    pl.add_err(cents+10,y_cilc_cross,yerr=ey_cilc_cross,marker="o",ls="none",label='constrained')
    pl.add_err(cents,y_silc_auto,yerr=ey_silc_auto,marker="d",ls="none",label='standard')
    pl.add_err(cents+10,y_cilc_auto,yerr=ey_cilc_auto,marker="d",ls="none",label='constrained')
    pl.done(io.dout_dir+"y_cross.png")



