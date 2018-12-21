from __future__ import print_function
# import matplotlib
# matplotlib.use('Agg')
from orphics import maps,io,cosmology,lensing,stats,mpi
from pixell import enmap,lensing as enlensing,utils
from enlib import bench
import numpy as np
import os,sys
from szar import foregrounds as fg
from tilec import utils as tutils,covtools,ilc

"""
Notes:
1. eigpow(-1) is bad for analytic (discontinuity at high ell)
2. large low ell scatter when including Planck comes from noisy cross-spectra due to only having 2 splits
3. for low fourier pixel density (low res, low ellmax), the fft based anisotropic binning can have a lot of ringing
4. lpass = False didn't fix new scatter
"""

aseed = 3
planck_autos = False
lensing = False # no need for lensing for initial tests
invert = True # 40x speedup from precalulated linalg.inv instead of linalg.solve
lpass = False # whether to remove modes below lmin in each of the arrays
# cov
analytic = False # whether to use an analytic or empirical covmat
atmosphere = True # whether to ignore atmosphere parameters
noise_isotropic = False # whether to average the noise spectra the same way as signal
debug_noise = False # whether to make debug plots of noise
# foregrounds
fgs = True # whether to include any foregrounds
dust = False # whether to include dust/CIB
ycibcorr = False # whether tSZ/CIB are correlated
# analysis
lmax = 6000
px = (1.0*10000./(lmax+500.))
# sims
nsims = 5
# signal cov
bin_width = 80 # this parameter seems to be important and cause unpredictable noise
kind = 0 # higher order interpolation breaks covariance
# noise cov
dfact=(16,16)

width = 35.
height = 15.


beams,freqs,noises,lknees,alphas,nsplits,lmins,lmaxs = np.loadtxt("input/simple_sim.txt",unpack=True)
if not(fgs):
    components = []
else:
    components = ['tsz','cib'] if dust else ['tsz']
comm,rank,my_tasks = mpi.distribute(nsims)

def process(kmaps,ellmax=None,dtype=np.complex128):
    ellmax = lmax if ellmax is None else ellmax
    kout = enmap.zeros((Ny,Nx),wcs,dtype=dtype).reshape(-1)
    kout[modlmap.reshape(-1)<lmax1] = np.nan_to_num(kmaps.copy())
    kout = enmap.enmap(kout.reshape((Ny,Nx)),wcs)
    # io.plot_img(fc.ifft(kout).real)
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
    def __init__(self,shape,wcs,beams,freqs,noises,lknees,alphas,nsplits,pss,nu0,lmins,lmaxs,theory=None):
        if not(atmosphere):
            lknees = [0.]*len(freqs)
            alphas = [1.]*len(freqs)
        self.nu0 = nu0
        self.freqs = freqs
        self.lmins = lmins
        self.lmaxs = lmaxs
        self.modlmap = enmap.modlmap(shape,wcs)
        lmax = self.modlmap.max()
        if theory is None: theory = cosmology.default_theory()
        ells = np.arange(0,lmax,1)
        cltt = theory.uCl('TT',ells) if lensing else theory.lCl('TT',ells)
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
        self.ebeams = []
        self.ps_noises = []
        self.eps_noises = []
        for array in self.arrays:
            ps_noise = cosmology.noise_func(ells,0.,noises[array],lknee=lknees[array],alpha=alphas[array])
            if lpass: ps_noise[ells<lmins[array]] = 0
            self.ps_noises.append(maps.interp(ells,ps_noise.copy())(self.modlmap))
            self.eps_noises.append(ps_noise.copy())
            self.ngens.append( maps.MapGen(shape[-2:],wcs,ps_noise[None,None]*nsplits[array]) )
            self.kbeams.append( maps.gauss_beam(self.modlmap,beams[array]) )
            self.lbeams.append( maps.gauss_beam(self.modlmap,beams[array])[self.modlmap<lmax1].reshape(-1) )
            self.ebeams.append( maps.gauss_beam(ells,beams[array]))
        self.ells = ells

    def get_corr(self,seed):
        fmap = self.fgen.get_map(seed=(aseed,self.kseed,seed),scalar=True)
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
        unlensed = self.cgen.get_map(seed=(aseed,self.cseed,seed))
        lensed = self._lens(unlensed,kappa) if lensing else unlensed
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
                noise = self.ngens[array].get_map(seed=(aseed,self.nseed,seed,split,array))
                observed[array].append(beamed+noise)
                noises[array].append(noise)
            observed[array] = enmap.enmap(np.stack(observed[array]),self.wcs)
            noises[array] = enmap.enmap(np.stack(noises[array]),self.wcs)
            if lpass:
                observed[array] = maps.filter_map(observed[array],maps.mask_kspace(self.shape,self.wcs,lmin=self.lmins[array]))             
                noises[array] = maps.filter_map(noises[array],maps.mask_kspace(self.shape,self.wcs,lmin=self.lmins[array]))             
        return observed,noises


shape,wcs = maps.rect_geometry(width_deg=width,height_deg=height,px_res_arcmin=px)
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
# pl = io.Plotter(xscale='log',yscale='log',scalefn=lambda x:x**2.)
# pl.add(ells,clss)
# pl._ax.set_xlim(2,1000)
# pl.done()
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

minell = maps.minimum_ell(tsim.shape,tsim.wcs)
bin_edges = np.arange(np.min(lmins),lmax-50,8*minell)
binner = stats.bin2D(modlmap,bin_edges)
cents = binner.centers

if analytic:
    Cov = ilc.build_analytic_cov(tsim.modlmap,theory.lCl('TT',tsim.modlmap),
                                 ffuncs,tsim.freqs,tsim.kbeams,
                                 tsim.ps_noises,lmins=lmins,lmaxs=lmaxs,verbose=True)
    Cov = Cov[:,:,modlmap<lmax1].reshape((narrays,narrays,modlmap[modlmap<lmax1].size))

s = stats.Stats(comm)

for task in my_tasks:
    with bench.show("sim gen"):
        isim,isimnoise = tsim.get_sim(task)
    with bench.show("ffts"):
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

    with bench.show("empirical cov"):
        if not(analytic):
            Scov = np.zeros((narrays,narrays,nells))
            Ncov = np.zeros((narrays,narrays,nells))
            for aindex1 in range(narrays):
                for aindex2 in range(aindex1,narrays) :
                    if auto_for_cross_covariance and aindex1!=aindex2:
                        scov = fc.f2power(ikmaps[aindex1],ikmaps[aindex2])
                        ncov = None
                    else:
                        scov,ncov,autos = tutils.ncalc(iksplits,aindex1,aindex2,fc)
                    if planck_autos and (tsim.nsplits[aindex1]<4) and (aindex1==aindex2): # if Planck
                        scov = autos
                        ncov = None
                    if aindex1==aindex2:
                        if ncov is None:
                            dncov = None
                        elif noise_isotropic:
                            dncov = covtools.signal_average(ncov,bin_width=bin_width,kind=kind,lmin=lmins[aindex1])
                        else:
                            dncov,_,_ = covtools.noise_average(ncov,dfact=dfact,
                                                               radial_fit=True if (tsim.nsplits[aindex1]==4 and atmosphere) else False,lmax=lmax,
                                                               wnoise_annulus=500,
                                                               lmin = 300,
                                                               bin_annulus=bin_width)
                    else:
                        dncov = None
                    dscov = covtools.signal_average(scov,bin_width=bin_width,kind=kind,lmin=max(lmins[aindex1],lmins[aindex2])) # need to check this is not zero # ((a,inf),(inf,inf))  doesn't allow the first element to be used, so allow for cross-covariance from non informative
                    #io.plot_img(maps.ftrans(dscov),aspect='auto')

                    if dncov is None: dncov = np.zeros(dscov.shape)
                    if debug_noise:
                        if aindex1==aindex2:
                            io.plot_img(maps.ftrans(scov),aspect='auto')
                            io.plot_img(maps.ftrans(dscov),aspect='auto')
                            io.plot_img(maps.ftrans(ncov),aspect='auto')
                            io.plot_img(maps.ftrans(dncov),aspect='auto')
                            io.plot_img(maps.ftrans(tsim.ps_noises[aindex1]),aspect='auto')
                    if aindex1==aindex2:
                        dncov[modlmap<lmins[aindex1]] = np.inf
                        dncov[modlmap>lmaxs[aindex1]] = np.inf
                    Scov[aindex1,aindex2] = dscov[modlmap<lmax1].reshape(-1).copy()
                    Ncov[aindex1,aindex2] = dncov[modlmap<lmax1].reshape(-1).copy()
                    if aindex1!=aindex2: Scov[aindex2,aindex1] = Scov[aindex1,aindex2].copy()

            Cov = Scov + Ncov

    ls = modlmap[modlmap<lmax1].reshape(-1)
    with bench.show("init ILC"):
        hilc = ilc.HILC(ls,np.array(tsim.lbeams),Cov,responses={'tsz':yresponses,'cmb':cresponses},chunks=1,invert=invert)

    with bench.show("more ffts"):
        ilensed = tsim.lensed
        _,iklensed,_ = fc.power2d(ilensed)
        iy = tsim.y
        _,iky,_ = fc.power2d(iy)
    with bench.show("ilc"):
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
    for array in tsim.arrays:
        pl.add(tsim.ells,tsim.eps_noises[array]/tsim.ebeams[array]**2.,alpha=0.3)
    icltt = binner.bin(theory.lCl('TT',modlmap))[1]
    pl.add(cents,icltt,marker="x",ls="none",color='k')
    pl.add_err(cents-5,cmb_silc_cross,yerr=ecmb_silc_cross,marker="o",ls="none",label='standard cross')
    pl.add_err(cents-10,cmb_cilc_cross,yerr=ecmb_cilc_cross,marker="o",ls="none",label='constrained  cross')
    pl.add_err(cents+5,cmb_silc_auto,yerr=ecmb_silc_auto,marker="x",ls="none",label='standard - noise')
    pl.add_err(cents+10,cmb_cilc_auto,yerr=ecmb_cilc_auto,marker="x",ls="none",label='constrained  - noise')
    # pl._ax.set_ylim(2e3,9e4)
    pl._ax.set_ylim(2e-1,9e4)
    pl._ax.set_xlim(0,lmax)
    # pl._ax.set_xlim(0,2500)
    pl.legend(loc='lower left')
    pl.done(io.dout_dir+"cmb_cross.png")

    io.save_cols("cmb_results.txt",(cents,cmb_silc_cross,ecmb_silc_cross,cmb_cilc_cross,ecmb_cilc_cross,cmb_silc_auto,ecmb_silc_auto,cmb_cilc_auto,ecmb_cilc_auto))


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


    io.save_cols("y_results.txt",(cents,y_silc_cross,ey_silc_cross,y_cilc_cross,ey_cilc_cross,y_silc_auto,ey_silc_auto,y_cilc_auto,ey_cilc_auto))
    

