from __future__ import print_function
# import matplotlib
# matplotlib.use('Agg')
from orphics import maps,io,cosmology,lensing,stats,mpi
from pixell import enmap,lensing as enlensing,utils
import numpy as np
import os,sys
from szar import foregrounds as fg
from tilec import utils as tutils,covtools,ilc

"""
Notes:
1. eigpow(-1) is bad for analytic (discontinuity at high ell)
"""

# cov
analytic = False
noise_isotropic = False
debug_noise = False
# foregrounds
fgs = True
dust = False
ycibcorr = False
# analysis
lmax = 6000
px = 1.0
# sims
nsims = 5
# signal cov
bin_width = 80 # this parameter seems to be important and cause unpredictable noise
kind = 0 # higher order interpolation breaks covariance
# noise cov
dfact=(16,16)


beams,freqs,noises,lknees,alphas,nsplits,lmins,lmaxs = np.loadtxt("input/simple_sim.txt",unpack=True)
if not(fgs):
    components = []
else:
    components = ['tsz','cib'] if dust else ['tsz']
comm,rank,my_tasks = mpi.distribute(nsims)

def process(kmaps,ellmax=None):
    ellmax = lmax if ellmax is None else ellmax
    kout = enmap.zeros((Ny,Nx),wcs,dtype=np.complex128).reshape(-1)
    kout[modlmap.reshape(-1)<lmax1] = np.nan_to_num(kmaps.copy())
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
    def __init__(self,shape,wcs,beams,freqs,noises,lknees,alphas,nsplits,pss,nu0,lmins,lmaxs):
        self.nu0 = nu0
        self.freqs = freqs
        self.lmins = lmins
        self.lmaxs = lmaxs
        self.modlmap = enmap.modlmap(shape,wcs)
        lmax = self.modlmap.max()
        theory = cosmology.default_theory()
        ells = np.arange(0,lmax,1)
        cltt = theory.uCl('TT',ells)
        self.cseed = 0
        self.kseed = 1
        self.nseed = 2
        self.fgen = maps.MapGen((ncomp,)+shape[-2:],wcs,pss)
        self.cgen = maps.MapGen(shape[-2:],wcs,cltt[None,None])
        self.shape, self.wcs = shape,wcs
        self.arrays = range(len(freqs))
        self.nsplits = nsplits
        self.ngens = []
        self.kbeams = []
        self.lbeams = []
        self.ps_noises = []
        for array in self.arrays:
            ps_noise = cosmology.noise_func(ells,0.,noises[array],lknee=lknees[array],alpha=alphas[array])
            # ps_noise[ells<lmins[array]] = 0
            # ps_noise[ells>lmaxs[array]] = 0
            self.ps_noises.append(maps.interp(ells,ps_noise.copy())(self.modlmap))
            self.ngens.append( maps.MapGen(shape[-2:],wcs,ps_noise[None,None]*nsplits[array]) )
            self.kbeams.append( maps.gauss_beam(self.modlmap,beams[array]) )
            self.lbeams.append( maps.gauss_beam(self.modlmap,beams[array])[self.modlmap<lmax1].reshape(-1) )

    def get_corr(self,seed):
        fmap = self.fgen.get_map(seed=(self.kseed,seed),scalar=True)
        return fmap

    def _lens(self,unlensed,kappa,lens_order=5):
        self.kappa = kappa
        alpha = lensing.alpha_from_kappa(kappa,posmap=enmap.posmap(self.shape,self.wcs))
        lensed = enlensing.displace_map(unlensed, alpha, order=lens_order)
        return lensed

    def get_sim(self,seed):
        ret = self.get_corr(seed)
        if dust:
            kappa,tsz,cib = ret
        else:
            kappa,tsz = ret
        unlensed = self.cgen.get_map(seed=(self.cseed,seed))
        lensed = self._lens(unlensed,kappa)
        self.lensed = lensed.copy()
        tcmb = 2.726e6
        self.y = tsz.copy()/tcmb/fg.ffunc(self.nu0)
        observed = []
        noises = []
        for array in self.arrays:
            scaled_tsz = tsz * fg.ffunc(self.freqs[array]) / fg.ffunc(self.nu0) if fgs else 0.
            if dust:
                scaled_cib = cib * fg.cib_nu(self.freqs[array]) / fg.cib_nu(self.nu0) if fgs else 0.
            else:
                scaled_cib = 0.
            sky = lensed + scaled_tsz + scaled_cib
            beamed = maps.filter_map(sky,self.kbeams[array])
            observed.append([])
            noises.append([])
            for split in range(self.nsplits[array]):
                noise = self.ngens[array].get_map(seed=(self.nseed,seed,split,array))
                observed[array].append(beamed+noise)
                noises[array].append(noise)
            observed[array] = enmap.enmap(np.stack(observed[array]),self.wcs)
            noises[array] = enmap.enmap(np.stack(noises[array]),self.wcs)
        return observed,noises


shape,wcs = maps.rect_geometry(width_deg=35.,height_deg=15.,px_res_arcmin=px)
minell = maps.minimum_ell(shape,wcs)
lmax1 = lmax-minell

ells = np.arange(0,lmax,1)
theory = cosmology.default_theory()

nu0 = 150.
if dust: cibkcorr = fg.kappa_cib_corrcoeff(ells)
ycorr = fg.y_kappa_corrcoeff(ells)

cltt = theory.lCl('tt',ells)
clkk = theory.gCl('kk',ells)
clss = fg.power_tsz(ells,nu0)
if dust: 
    clsc = fg.power_tsz_cib(ells,nu0) if ycibcorr else clss*0.
    clcc = fg.power_cibc(ells,nu0)
    clkc = cibkcorr * np.sqrt(clkk*clcc)
clks = ycorr*np.sqrt(clkk*clss)

ffuncs = {}
ffuncs['tsz'] = lambda ells,nu1,nu2: fg.power_tsz(ells,nu1,nu2)
if dust: ffuncs['cib'] = lambda ells,nu1,nu2: fg.power_cibc(ells,nu1,nu2)

ncomp = 3 if dust else 2
ps = np.zeros((ncomp,ncomp,ells.size))
ps[0,0] = clkk
ps[1,1] = clss
ps[0,1] = ps[1,0] = clks
if dust:
    ps[2,2] = clcc
    ps[1,2] = ps[2,1] = clsc
    ps[0,2] = ps[2,0] = clkc

tsim = TSimulator(shape,wcs,beams,freqs,noises,lknees,alphas,nsplits.astype(np.int),ps,nu0,lmins=lmins,lmaxs=lmaxs)
modlmap = tsim.modlmap
fc = maps.FourierCalc(tsim.shape,tsim.wcs)
narrays = len(tsim.arrays)
iells = modlmap[modlmap<lmax1].reshape(-1) # unraveled disk
nells = iells.size
Ny,Nx = shape[-2:]
tcmb = 2.726e6
yresponses = fg.ffunc(tsim.freqs)*tcmb
cresponses = yresponses*0.+1.

bin_edges = np.arange(200,lmax-50,80)
binner = stats.bin2D(modlmap,bin_edges)
cents = binner.centers

if analytic:
    Cov = ilc.build_analytic_cov(tsim.modlmap,theory.lCl('TT',tsim.modlmap),
                                 ffuncs,tsim.freqs,tsim.kbeams,
                                 tsim.ps_noises,lmins=lmins,lmaxs=lmaxs,verbose=True)
    Cov = Cov[:,:,modlmap<lmax1].reshape((narrays,narrays,modlmap[modlmap<lmax1].size))
    # iCov = np.rollaxis(Cov,2)
    # icinv = np.linalg.inv(iCov)

s = stats.Stats(comm)

for task in my_tasks:
    isim,isimnoise = tsim.get_sim(task)
    ikmaps = []
    inkmaps = []
    iksplits = []
    for array in tsim.arrays:
        iksplits.append([])
        for split in range(tsim.nsplits[array]):
            _,_,ksplit = fc.power2d(isim[array][split])
            iksplits[array].append(ksplit.copy())
        iksplits[array] = enmap.enmap(np.stack(iksplits[array]),tsim.wcs)
            
        kcoadd = sum(iksplits[array])/tsim.nsplits[array]
        ncoadd = sum(isimnoise[array])/tsim.nsplits[array]
        _,_,kncoadd = fc.power2d(ncoadd)
        ikmaps.append(  kcoadd.copy())
        inkmaps.append(  kncoadd.copy())

    if not(analytic):
        Scov = np.zeros((narrays,narrays,nells))
        Ncov = np.zeros((narrays,narrays,nells))
        for aindex1 in range(narrays):
            for aindex2 in range(aindex1,narrays) :
                scov,ncov,autos = tutils.ncalc(iksplits,aindex1,aindex2,fc)
                dscov = covtools.signal_average(scov,bin_width=bin_width,kind=kind) # need to check this is not zero
                if noise_isotropic:
                    dncov = covtools.signal_average(ncov,bin_width=bin_width,kind=kind) if (aindex1==aindex2)  else 0.
                else:
                    dncov,_,_ = covtools.noise_average(ncov,dfact=dfact,
                                                       radial_fit=True if tsim.nsplits[aindex1]==4 else False,lmax=lmax,
                                                       wnoise_annulus=500,
                                                       bin_annulus=bin_width) if (aindex1==aindex2)  else (0.,None,None)
                if debug_noise:
                    if aindex1==aindex2:
                        io.plot_img(maps.ftrans(scov),aspect='auto')
                        io.plot_img(maps.ftrans(dscov),aspect='auto')
                        io.plot_img(maps.ftrans(ncov),aspect='auto')
                        io.plot_img(maps.ftrans(dncov),aspect='auto')
                        io.plot_img(maps.ftrans(tsim.ps_noises[aindex1]),aspect='auto')
                dncov = np.nan_to_num(dncov)
                dscov = np.nan_to_num(dscov)
                if aindex1==aindex2:
                    dncov[modlmap<lmins[aindex1]] = np.inf
                    dncov[modlmap>lmaxs[aindex1]] = np.inf
                Scov[aindex1,aindex2] = dscov[modlmap<lmax1].reshape(-1).copy()
                Ncov[aindex1,aindex2] = dncov[modlmap<lmax1].reshape(-1).copy() if (aindex1==aindex2)  else 0.
                if aindex1!=aindex2:
                    Scov[aindex2,aindex1] = Scov[aindex1,aindex2].copy()
                    Ncov[aindex2,aindex1] = Ncov[aindex1,aindex2].copy()
                
        Cov = Scov + Ncov
        # iCov = np.rollaxis(Cov,2)
        # icinv = np.linalg.inv(iCov)
        
    ls = modlmap[modlmap<lmax1].reshape(-1)
    hilc = ilc.HILC(ls,np.array(tsim.lbeams),Cov,responses={'tsz':yresponses,'cmb':cresponses},chunks=1)
        
    ilensed = tsim.lensed
    _,iklensed,_ = fc.power2d(ilensed)
    iy = tsim.y
    _,iky,_ = fc.power2d(iy)
    ikmaps = np.stack(ikmaps)
    ikmaps[:,modlmap>lmax1] = 0
    inkmaps = np.stack(inkmaps)
    inkmaps[:,modlmap>lmax1] = 0
    kmaps = ikmaps.reshape((narrays,Ny*Nx))[:,modlmap.reshape(-1)<lmax1]
    nkmaps = inkmaps.reshape((narrays,Ny*Nx))[:,modlmap.reshape(-1)<lmax1]
    iksilc = process(hilc.standard_map(kmaps,"tsz"))
    inksilc = process(hilc.standard_map(nkmaps,"tsz"))
    compute(iksilc,iky,"y_silc_cross")
    ncompute(iksilc,inksilc,"y_silc_auto")
    iksilc = process(hilc.constrained_map(kmaps,"tsz","cmb"))
    inksilc = process(hilc.constrained_map(nkmaps,"tsz","cmb"))
    compute(iksilc,iky,"y_cilc_cross")
    ncompute(iksilc,inksilc,"y_cilc_auto")
    
    iksilc = process(hilc.standard_map(kmaps,"cmb"))
    inksilc = process(hilc.standard_map(nkmaps,"cmb"))
    compute(iksilc,iklensed,"cmb_silc_cross")
    ncompute(iksilc,inksilc,"cmb_silc_auto")
    iksilc = process(hilc.constrained_map(kmaps,"cmb","tsz"))
    inksilc = process(hilc.constrained_map(nkmaps,"cmb","tsz"))
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
    pl.add_err(cents-5,cmb_silc_cross,yerr=ecmb_silc_cross,marker="o",ls="none",label='standard cross')
    pl.add_err(cents-10,cmb_cilc_cross,yerr=ecmb_cilc_cross,marker="o",ls="none",label='constrained  cross')
    pl.add_err(cents+5,cmb_silc_auto,yerr=ecmb_silc_auto,marker="x",ls="none",label='standard - noise')
    pl.add_err(cents+10,cmb_cilc_auto,yerr=ecmb_cilc_auto,marker="x",ls="none",label='constrained  - noise')
    pl._ax.set_ylim(1e0,5e4)
    pl.done(io.dout_dir+"cmb_cross.png")


    pl = io.Plotter(scalefn=lambda x: x**2,xlabel='l',ylabel='D',yscale='log')
    pl.add(ells,clyy)
    pl.add_err(cents-5,y_silc_cross,yerr=ey_silc_cross,marker="o",ls="none",label='standard cross')
    pl.add_err(cents-10,y_cilc_cross,yerr=ey_cilc_cross,marker="o",ls="none",label='constrained cross')
    pl.add_err(cents+5,y_silc_auto,yerr=ey_silc_auto,marker="x",ls="none",label='standard - noise')
    pl.add_err(cents+10,y_cilc_auto,yerr=ey_cilc_auto,marker="x",ls="none",label='constrained - noise')
    # pl._ax.set_ylim(-1e-12,1e-12)
    # pl.hline()
    pl._ax.set_ylim(1e-13,5e-11)
    pl.done(io.dout_dir+"y_cross.png")



