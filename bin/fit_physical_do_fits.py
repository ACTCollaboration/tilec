from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap
import numpy as np
import os,sys
import pickle
from szar import foregrounds as fgs
import soapack.interfaces as sints
from tilec import kspace,utils as tutils,ilc
from actsims import noise as simnoise
from enlib import bench

"""

We will create a covariance matrix (narrays,narrays,ellmax)
that describes what the power spectra of "residuals" are.
Residual is defined to be what is left over after subtracting
a fiducial lensed CMB and fiducial tSZ spectrum. This procedure
is aimed at producing Gaussian simulations whose total power
matches that of the data without having to model, for example,
the residual foreground power after a complicated source
subtraction procedure, and without having to, for example, add
back sources to make the residual easy to model (thus increasing
noise and bias in a lensing estimator).

We do this by taking the following spectra in the specified
ell ranges,

LFI-30/44 x LFI  20 < ell < 300
LFI-70    x LFI  20 < ell < 900
LFI-30/44 x HFI  20 < ell < 300
LFI-70    x HFI  20 < ell < 2000
LFI       x HFI  20 < ell < 300
LFI-30/44 x ACT  -- no residual --
LFI-70    x ACT  1000 < ell < 2000
HFI       x ACT  1000 < ell < 5000
HFI       x HFI  20 < ell < 3000
ACT       x ACT  1000 < ell < 8000




"""


def plaw(ells,a1,a2,e1,e2,w):
    p = w + a1*(ells/3000)**e1 + a2*(ells/3000)**e2
    return p

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("version", type=str,help='Version name.')
parser.add_argument("region", type=str,help='Region name.')
parser.add_argument("arrays", type=str,help='Comma separated list of array names. Array names map to a data specification in data.yml')
parser.add_argument("--mask-version", type=str,  default="padded_v1",help='Mask version')
parser.add_argument("--unsanitized-beam", action='store_true',help='Do not sanitize beam.')
lmax = 8101
args = parser.parse_args()

spath = sints.dconfig['actsims']['fg_res_path'] + "/"+ args.version + "_" + args.region +  "/"

try: os.makedirs(spath)
except: pass


aqids = args.arrays.split(',')
narrays = len(aqids)
qpairs = []
for qid1 in range(narrays):
    for qid2 in range(qid1,narrays):
        qpairs.append((aqids[qid1],aqids[qid2]))


"""
We will MPI parallelize over pairs of arrays. This wastes FFTs, but is more memory efficient. (Each job only
holds 2 arrays in memory).
"""
njobs = len(qpairs)
comm,rank,my_tasks = mpi.distribute(njobs)


mask = sints.get_act_mr3_crosslinked_mask(args.region,
                                          version=args.mask_version,
                                          kind='binary_apod')
shape,wcs = mask.shape,mask.wcs
modlmap = mask.modlmap()
aspecs = tutils.ASpecs().get_specs
region = args.region

fbeam = lambda qname,x: tutils.get_kbeam(qname,x,sanitize=not(args.unsanitized_beam),planck_pixwin=True)
nbin_edges = np.arange(20,8000,100)
nbinner = stats.bin2D(modlmap,nbin_edges)
ncents = nbinner.centers

cbin_edges = np.arange(20,8000,20)
cbinner = stats.bin2D(modlmap,cbin_edges)
fells = np.arange(lmax)

for task in my_tasks:
    qids = qpairs[task]
    qid1,qid2 = qids
    ncents,n1d = np.loadtxt("%sn1d_%s_%s.txt" % (spath,qid1,qid2),unpack=True)
    ncents,n1d1 = np.loadtxt("%sn1d_%s_%s.txt" % (spath,qid1,qid1),unpack=True)
    ncents,n1d2 = np.loadtxt("%sn1d_%s_%s.txt" % (spath,qid2,qid2),unpack=True)
    ccents,s1d = np.loadtxt("%ss1d_%s_%s.txt" % (spath,qid1,qid2),unpack=True)
    fbeam1 = lambda x: tutils.get_kbeam(qid1,x,sanitize=not(args.unsanitized_beam),planck_pixwin=True)
    fbeam2 = lambda x: tutils.get_kbeam(qid2,x,sanitize=not(args.unsanitized_beam),planck_pixwin=True)

    lmin,lmax,hybrid,radial,friend,f1,fgroup,wrfit = aspecs(qid1)
    lmin,lmax,hybrid,radial,friend,f2,fgroup,wrfit = aspecs(qid2)

    ctheory = ilc.CTheory(ncents)
    stheory = ilc.CTheory(ccents)
    cltt = ctheory.get_theory_cls(f1,f2,a_cmb=1,a_gal=0.8)
    pl = io.Plotter(xyscale='linlog',scalefn=lambda x: x**2./2./np.pi,xlabel='l',ylabel='D')
    pl.add(ncents,cltt,color='k',lw=3)
    pl.add(ncents,n1d/fbeam1(ncents)/fbeam2(ncents))
    pl._ax.set_xlim(2,8000)
    pl._ax.set_ylim(1,1e8)
    pl.done("%sn1d_%s_%s.png" % (spath,qid1,qid2),verbose=False)


    pl = io.Plotter(xyscale='linlog',scalefn=lambda x: x**2./2./np.pi,xlabel='l',ylabel='D')
    if tutils.is_planck(qid1) and tutils.is_planck(qid2): pl.add(ncents,cltt,color='k',lw=3) # unblind only if both planck
    pl.add(ccents,s1d/fbeam1(ccents)/fbeam2(ccents))
    pl.add(ncents,n1d/fbeam1(ncents)/fbeam2(ncents))
    pl._ax.set_xlim(2,8000)
    pl._ax.set_ylim(1,1e8)
    pl.done("%ss1d_%s_%s.png" % (spath,qid1,qid2),verbose=False)


    res = s1d/fbeam1(ccents)/fbeam2(ccents) - stheory.get_theory_cls(f1,f2,a_cmb=1,a_gal=0,a_cibp=0,a_cibc=0,a_radps=0,a_ksz=0,a_tsz=1)

    if tutils.is_planck(qid1) and tutils.is_planck(qid2): 
        flmin = 20
    else:
        flmin = 1000
        
    if not(tutils.is_planck(qid1)) and not(tutils.is_planck(qid2)): 
        flmax = 5000
    elif tutils.is_hfi(qid1) and tutils.is_hfi(qid2):
        flmax = 3000
    elif (not(tutils.is_planck(qid1)) and tutils.is_hfi(qid2)) or (not(tutils.is_planck(qid2)) and tutils.is_hfi(qid1)):
        flmax = 4000
    elif (qid1=='p03' or qid2=='p03') and (qid1 not in ['p01','p02']) and (qid2 not in ['p01','p02']):
        flmax = 1000
    else:
        flmax = 300
    
    if flmax<=flmin: 
        io.save_cols("%sfgcov_%s_%s.txt" % (spath,qid1,qid2),(fells,fells*0))
        continue

    print("Rank %d doing task %d for array %s x %s with lmin %d and lmax %d ..." % (rank,task,qids[0],qids[1],flmin,flmax))

    # ERROR CALC
    c11 = stheory.get_theory_cls(f1,f1,a_cmb=1,a_gal=0.8) 
    n11 = maps.interp(ncents,n1d1)(ccents)/fbeam1(ccents)/fbeam1(ccents)
    c22 = stheory.get_theory_cls(f2,f2,a_cmb=1,a_gal=0.8) 
    n22 = maps.interp(ncents,n1d2)(ccents)/fbeam2(ccents)/fbeam2(ccents)
    c12 = stheory.get_theory_cls(f1,f2,a_cmb=1,a_gal=0.8) 
    n12 = maps.interp(ncents,n1d)(ccents)/fbeam1(ccents)/fbeam2(ccents)
    c11[~np.isfinite(c11)] = 0
    c12[~np.isfinite(c12)] = 0
    c22[~np.isfinite(c22)] = 0
    cbin_edges = np.arange(20,8000,20)
    LF = cosmology.LensForecast()
    LF.loadGenericCls("11",ccents,c11,ellsNls=ccents,Nls=n11)        
    LF.loadGenericCls("22",ccents,c22,ellsNls=ccents,Nls=n22)        
    LF.loadGenericCls("12",ccents,c12,ellsNls=ccents,Nls=n12)         
    if region=='deep56':
        fsky = 500./41252.
    if region=='boss':
        fsky = 1700/41252.
    _,errs = LF.sn(cbin_edges,fsky,'12')

    ###

    sel = np.logical_and(ccents>flmin,ccents<flmax)

    ffunc = plaw #(ells,a1,a2,e1,e2,w)
    from scipy.optimize import curve_fit
    try:
        #popt,pcov = curve_fit(ffunc,ccents[sel],res[sel]*ccents[sel]**2.,p0=[1,1,-2,2,1e2],bounds=([0,0,-np.inf,0,0],[np.inf,np.inf,0,2,np.inf]),absolute_sigma=True,sigma=errs[sel]*ccents[sel]**2.)
        popt,pcov = curve_fit(ffunc,ccents[sel],res[sel]*ccents[sel]**2.,p0=[1,1,-1,1,1e2],bounds=([0,0,-5,0,0],[np.inf,np.inf,0,2,np.inf]),absolute_sigma=True,sigma=errs[sel]*ccents[sel]**2.)
        pfit = lambda x: ffunc(x,popt[0],popt[1],popt[2],popt[3],popt[4])/x**2
        print("a1:", popt[0], "a2:", popt[1], "e1:",popt[2], "e2:",popt[3], "w:" ,popt[4])
    except:
        pfit = lambda x: x*0


    pl = io.Plotter(xyscale='linlog',scalefn=lambda x: x**2./2./np.pi,xlabel='l',ylabel='D')
    if tutils.is_planck(qid1) and tutils.is_planck(qid2): pl.add(ccents,stheory.get_theory_cls(f1,f2,a_cmb=0,a_gal=0.8,a_tsz=0),color='k',lw=3) # unblind only if both planck
    pl.add_err(ccents,res,yerr=errs,marker="o",ls="none")
    pl.add(ccents[sel],pfit(ccents[sel]),ls='--')
    pl._ax.set_xlim(flmin,flmax)
    pl._ax.set_ylim(0.1,1e8)
    pl.done("%sres_%s_%s_%s.png" % (spath,region,qid1,qid2),verbose=False)


    pl = io.Plotter(xyscale='linlin',scalefn=lambda x: x**2./2./np.pi,xlabel='l',ylabel='D')
    if tutils.is_planck(qid1) and tutils.is_planck(qid2): pl.add(ccents,stheory.get_theory_cls(f1,f2,a_cmb=0,a_gal=0.8,a_tsz=0),color='k',lw=3) # unblind only if both planck
    pl.add_err(ccents,res,yerr=errs,marker="o",ls="none")
    pl.add(cbin_edges,pfit(cbin_edges),ls='--')
    pl._ax.set_xlim(flmin,flmax)
    pl._ax.set_ylim(-10,5e2)
    pl.hline(y=0)
    pl.done("%sreslin_%s_%s_%s.png" % (spath,region,qid1,qid2),verbose=False)

    
    scls = pfit(fells)
    scls[fells<20] = 0
    if not(tutils.is_planck(qid1)) or not(tutils.is_planck(qid2)): scls[fells<500] = 0
    #if tutils.is_lfi(qid1) or tutils.is_lfi(qid2): scls[fells>2000] = 0
    #if tutils.is_hfi(qid1) or tutils.is_hfi(qid2): scls[fells>4000] = 0

    pl = io.Plotter(xyscale='linlog',scalefn=lambda x: x**2./2./np.pi,xlabel='l',ylabel='D')
    pl.add(ncents,n1d)
    pl.add(ncents,n1d1)
    pl.add(ncents,n1d2)
    pl.add(fells,scls*fbeam1(fells)*fbeam2(fells),ls='--')
    pl.add(fells,scls,ls=':')
    # pl._ax.set_xlim(flmin,flmax)
    # pl._ax.set_ylim(-10,5e2)
    pl.done("%sresfull_%s_%s_%s.png" % (spath,region,qid1,qid2),verbose=False)


    io.save_cols("%sfgcov_%s_%s.txt" % (spath,qid1,qid2),(fells,scls))
