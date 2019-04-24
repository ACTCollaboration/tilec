"""
Chunked ILC simulator
Simulates arrays specified in input/simple_sim.txt (check that first)!

"""

from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from orphics import maps,io,cosmology,lensing,stats,mpi
from pixell import enmap,lensing as enlensing,utils
from enlib import bench
import numpy as np
import os,sys
from szar import foregrounds as fg
from tilec import utils as tutils,covtools,ilc,fg as tfg,kspace

"""
Notes:
1. eigpow(-1) is bad for analytic (discontinuity at high ell)
2. large low ell scatter when including Planck comes from noisy cross-spectra due to only having 2 splits
3. for low fourier pixel density (low res, low ellmax), the fft based anisotropic binning can have a lot of ringing
4. lpass = False didn't fix new scatter
"""

multi = False # whether to use Colin's multi-component code instead of Mat's less general max-1-component-deproject code
chunk_size = 20000 # number of fourier pixels in each chunk; lower number means lower peak memory usage but slower run
aseed = 3 # the overall seed for the analysis
lensing = False # whether to lens the CMB ; no need for lensing for initial tests
invert = True # whether to pre-invert the covmat;  40x speedup from precalulated linalg.inv instead of linalg.solve
lpass = False # whether to remove modes below lmin in each of the simulated arrays; default False
# cov
analytic = False # whether to use an analytic or empirical covmat; analytic=True means not real ILC!
atmosphere = True # whether to ignore atmosphere parameters in simple_sim.txt
noise_isotropic = False # whether to average the noise spectra the same way as signal; i.e. always uses radial binning, not the hybrid tILe-C (TM) binning
debug_noise = False # whether to make debug plots of noise
# foregrounds
fgs = True # whether to include any foregrounds at all (master switch for foregrounds)
dust = False # whether to include dust/CIB
ycibcorr = False # whether tSZ/CIB are correlated
# analysis
#lmax = 6000
#px = (1.0*10000./(lmax+500.))
lmax = 2000 # Overall low-pass filter for the analysis; set to large number to not apply one at all
px = 1.5 # resolution in arcminutes of the sims
# sims
nsims = 10 # number of sims
# signal cov
bin_width = 80 # the width in ell of radial bins for signal averaging
kind = 0 # order of interpolation for signal binning; higher order interpolation breaks covariance
# noise cov
dfact=(16,16) # downsample factor in ly and lx direction for noise

# simulated patch dimensions in degrees ; all sims are periodic
width = 15.
height = 15.




beams,freqs,noises,lknees,alphas,nsplits,lmins,lmaxs = np.loadtxt("input/simple_sim.txt",unpack=True)
if not(fgs):
    components = []
else:
    components = ['tsz','cib'] if dust else ['tsz']
comm,rank,my_tasks = mpi.distribute(nsims)

def compute(ik1,ik2,tag):
    kmap = ik1.reshape((Ny,Nx))
    # imap = enmap.enmap(fc.ifft(kmap).real,wcs)
    # io.hplot(imap,tag)
    pcross2d = fc.f2power(ik1.reshape((Ny,Nx)),ik2.reshape((Ny,Nx)))
    cents,p1d = binner.bin(pcross2d)
    s.add_to_stats(tag,p1d.copy())

def ncompute(ik,nk,tag):
    pauto2d = fc.f2power(ik.reshape((Ny,Nx)),ik.reshape((Ny,Nx))) - fc.f2power(nk.reshape((Ny,Nx)),nk.reshape((Ny,Nx)))
    cents,p1d = binner.bin(pauto2d)
    s.add_to_stats(tag,p1d.copy())
    



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

tsim = maps.SplitSimulator(shape,wcs,beams,freqs,noises,nsplits.astype(np.int),lknees,alphas,ps,nu0,lmins=lmins,lmaxs=lmaxs,atmosphere=atmosphere,lensing=lensing,dust=dust,do_fgs=fgs,lpass=lpass,aseed=aseed)

modlmap = tsim.modlmap
fc = maps.FourierCalc(tsim.shape,tsim.wcs)
narrays = len(tsim.arrays)
iells = modlmap[modlmap<lmax1].reshape(-1) # unraveled disk
nells = iells.size
Ny,Nx = shape[-2:]
tcmb = 2.726e6
yresponses = tfg.get_mix(tsim.freqs, "tSZ")
cresponses = tfg.get_mix(tsim.freqs, "CMB")

minell = maps.minimum_ell(tsim.shape,tsim.wcs)
bin_edges = np.arange(np.min(lmins),lmax-50,8*minell)
binner = stats.bin2D(modlmap,bin_edges)
cents = binner.centers

# Only do hybrid treatment for arrays with more than 2 splits (ACT) -- now doing for Planck as well
anisotropic_pairs = []
if not(noise_isotropic):
    for aindex1 in range(narrays):
        nsplits1 = tsim.nsplits[aindex1]
        if True: anisotropic_pairs.append((aindex1,aindex1))


if analytic:
    Cov = ilc.build_analytic_cov(tsim.modlmap,theory.lCl('TT',tsim.modlmap),
                                 ffuncs,tsim.freqs,tsim.kbeams,
                                 tsim.ps_noises,lmins=lmins,lmaxs=lmaxs,verbose=True)
    Cov = Cov[:,:,modlmap<lmax1].reshape((narrays,narrays,modlmap[modlmap<lmax1].size))

s = stats.Stats(comm)
wins = mask = enmap.ones(tsim.shape[-2:],tsim.wcs)

covdict = {}
names = ["a%d" % i for i in range(narrays)]
def save_fn(tcov,a1,a2): covdict[a1+"_"+a2] = tcov.copy()

for task in my_tasks:
    with bench.show("sim gen"):
        isim,isimnoise = tsim.get_sim(task)
    with bench.show("ffts"):
        iksplits = []
        ikmaps = []
        inkmaps = []
        for array in tsim.arrays:
            splits = isim[array]
            noise_splits = isimnoise[array]
            ksplits,kcoadd = kspace.process_splits(splits,wins=wins,mask=mask)
            _,kncoadd = kspace.process_splits(noise_splits,wins=wins,mask=mask,skip_splits=True)
            iksplits.append(  ksplits.copy())
            ikmaps.append(  kcoadd.copy())
            inkmaps.append(  kncoadd.copy())

    ls = modlmap[modlmap<lmax1].reshape(-1)
    with bench.show("empirical cov"):
        if not(analytic):
            atmospheres = [tsim.nsplits[array]>2 for array in range(narrays)]
            ilc.build_empirical_cov(names,iksplits,ikmaps,wins,mask,lmins,lmaxs,
                                          anisotropic_pairs,save_fn,
                                          signal_bin_width=bin_width,
                                          signal_interp_order=kind,
                                          dfact=dfact,
                                          rfit_lmaxes=None,
                                          rfit_wnoise_width=250,
                                          rfit_lmin=300,
                                          rfit_bin_width=None,debug_plots_loc='./')

    Cov = covdict
    print(Cov)
    for key in Cov.keys():
        io.plot_img(enmap.enmap(np.fft.fftshift(np.log10(Cov[key])),tsim.wcs),"%s.png" % key)
    with bench.show("more ffts"):
        ilensed = tsim.lensed
        _,iklensed,_ = fc.power2d(ilensed)
        iy = tsim.y
        _,iky,_ = fc.power2d(iy)
        
    ikmaps = np.stack(ikmaps)
    # ikmaps[:,modlmap>lmax1] = 0
    inkmaps = np.stack(inkmaps)
    # inkmaps[:,modlmap>lmax1] = 0
    ilcgen = ilc.chunked_ilc(modlmap,np.stack(tsim.kbeams),Cov,chunk_size,responses={'tsz':yresponses,'cmb':cresponses},invert=invert)

    yksilc = enmap.empty((Ny,Nx),wcs,dtype=np.complex128).reshape(-1)
    ynksilc = enmap.empty((Ny,Nx),wcs,dtype=np.complex128).reshape(-1)
    ykcilc = enmap.empty((Ny,Nx),wcs,dtype=np.complex128).reshape(-1)
    ynkcilc = enmap.empty((Ny,Nx),wcs,dtype=np.complex128).reshape(-1)

    cksilc = enmap.empty((Ny,Nx),wcs,dtype=np.complex128).reshape(-1)
    cnksilc = enmap.empty((Ny,Nx),wcs,dtype=np.complex128).reshape(-1)
    ckcilc = enmap.empty((Ny,Nx),wcs,dtype=np.complex128).reshape(-1)
    cnkcilc = enmap.empty((Ny,Nx),wcs,dtype=np.complex128).reshape(-1)
    
    kmaps = ikmaps.reshape((narrays,Ny*Nx))
    nkmaps = inkmaps.reshape((narrays,Ny*Nx))
    for chunknum,(hilc,selchunk) in enumerate(ilcgen):
        print("ILC on chunk ", chunknum+1, " / ",int(modlmap.size/chunk_size)+1," ...")
        yksilc[selchunk] = hilc.standard_map(kmaps[...,selchunk],"tsz")
        ynksilc[selchunk] = hilc.standard_map(nkmaps[...,selchunk],"tsz")
        if multi:
            ykcilc[selchunk] = hilc.multi_constrained_map(kmaps[...,selchunk],"tsz",["cmb"])
            ynkcilc[selchunk] = hilc.multi_constrained_map(nkmaps[...,selchunk],"tsz",["cmb"])
        else:
            ykcilc[selchunk] = hilc.constrained_map(kmaps[...,selchunk],"tsz","cmb")
            ynkcilc[selchunk] = hilc.constrained_map(nkmaps[...,selchunk],"tsz","cmb")

        cksilc[selchunk] = hilc.standard_map(kmaps[...,selchunk],"cmb")
        cnksilc[selchunk] = hilc.standard_map(nkmaps[...,selchunk],"cmb")
        if multi:
            ckcilc[selchunk] = hilc.multi_constrained_map(kmaps[...,selchunk],"cmb",["tsz"])
            cnkcilc[selchunk] = hilc.multi_constrained_map(nkmaps[...,selchunk],"cmb",["tsz"])
        else:
            ckcilc[selchunk] = hilc.constrained_map(kmaps[...,selchunk],"cmb","tsz")
            cnkcilc[selchunk] = hilc.constrained_map(nkmaps[...,selchunk],"cmb","tsz")
        
    with bench.show("ilc"):
        compute(yksilc,iky,"y_silc_cross")
        ncompute(yksilc,ynksilc,"y_silc_auto")
        compute(ykcilc,iky,"y_cilc_cross")
        ncompute(ykcilc,ynkcilc,"y_cilc_auto")
        compute(cksilc,iklensed,"cmb_silc_cross")
        ncompute(cksilc,cnksilc,"cmb_silc_auto")
        compute(ckcilc,iklensed,"cmb_cilc_cross")
        ncompute(ckcilc,cnkcilc,"cmb_cilc_auto")
    if rank==0: print ("Rank 0 done with task ", task+1, " / " , len(my_tasks))

s.get_stats()

if rank==0:
    # Cross of ILC CMB with input CMB
    cmb_silc_cross = s.stats["cmb_silc_cross"]['mean']
    cmb_cilc_cross = s.stats["cmb_cilc_cross"]['mean']
    # Cross of ILC y-map with input y-map
    y_silc_cross = s.stats["y_silc_cross"]['mean']
    y_cilc_cross = s.stats["y_cilc_cross"]['mean']
    # Errors on those
    ecmb_silc_cross = s.stats["cmb_silc_cross"]['errmean']
    ecmb_cilc_cross = s.stats["cmb_cilc_cross"]['errmean']
    ey_silc_cross = s.stats["y_silc_cross"]['errmean']
    ey_cilc_cross = s.stats["y_cilc_cross"]['errmean']
    # Auto of ILC CMB minus Auto of ILC CMB noise only
    cmb_silc_auto = s.stats["cmb_silc_auto"]['mean']
    cmb_cilc_auto = s.stats["cmb_cilc_auto"]['mean']
    # Auto of ILC y-map minus Auto of ILC y map noise only
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
    

