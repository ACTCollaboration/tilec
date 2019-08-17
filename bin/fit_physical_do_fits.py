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

def plaw(ells,a1,a2,a3,e1,e2,e3,b,ellp):
    return a1*(ells/ellp)**e1 + a2*(ells/ellp)**e2 + a3*(ells/ellp)**e3 + b

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("version", type=str,help='Version name.')
parser.add_argument("region", type=str,help='Region name.')
parser.add_argument("arrays", type=str,help='Comma separated list of array names. Array names map to a data specification in data.yml')
parser.add_argument("--mask-version", type=str,  default="padded_v1",help='Mask version')
parser.add_argument("--unsanitized-beam", action='store_true',help='Do not sanitize beam.')
parser.add_argument("--skip-inpainting", action='store_true',help='Do not inpaint.')

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
We will MPI parallelize over pairs of arrays. This wastes FFTs, but is much memory efficient. (Each job only
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


    res = s1d/fbeam1(ccents)/fbeam2(ccents) - stheory.get_theory_cls(f1,f2,a_cmb=1,a_gal=0,a_cibp=0,a_cibc=0,a_radps=0,a_ksz=0)

    if tutils.is_planck(qid1) and tutils.is_planck(qid2): 
        flmin = 20
    else:
        flmin = 1000
        
    if not(tutils.is_planck(qid1)) and not(tutils.is_planck(qid2)): 
        flmax = 8000
    elif tutils.is_hfi(qid1) and tutils.is_hfi(qid2):
        flmax = 4000
    elif (not(tutils.is_planck(qid1)) and tutils.is_hfi(qid2)) or (not(tutils.is_planck(qid2)) and tutils.is_hfi(qid1)):
        flmax = 5000
    elif (qid1=='p03' or qid2=='p03') and (qid1 not in ['p01','p02']) and (qid2 not in ['p01','p02']):
        flmax = 2000
    else:
        flmax = 300
    
    if flmax<flmin: continue
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

    #ffunc = plaw
    ffunc = lambda x,a_gal,exp_gal,a_radps,al_ps,a_cibp,a_cibc,a_ksz: stheory.get_theory_cls(f1,f2,a_cmb=0,a_gal=a_gal,exp_gal=exp_gal,a_cibp=a_cibp,a_cibc=a_cibc,a_radps=a_radps,a_ksz=a_ksz,a_tsz=0,al_ps=al_ps)[sel]

    from scipy.optimize import curve_fit
    #popt,pcov = curve_fit(ffunc,ccents[sel],res[sel],sigma=errs[sel],bounds=([0,0,0,0,0,-2,0,100],np.inf),absolute_sigma=True)
    #pfit = lambda x: ffunc(x,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7])

    popt,pcov = curve_fit(ffunc,ccents[sel],res[sel],p0=[0.8,-0.7,1,-0.5,1,1,1],sigma=errs[sel],bounds=([0,-1,0,-1,0,0,0],[np.inf,1,np.inf,1,np.inf,np.inf,np.inf]),absolute_sigma=True)
    pfit = lambda x: ffunc(x,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6])
    print("a_gal:%.2f, exp_gal:%.2f, a_ps:%.2f, exp_ps:%.2f, a_cibp:%.2f, a_cibc:%.2f, a_ksz:%.2f " %  (popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6]))


    pl = io.Plotter(xyscale='linlog',scalefn=lambda x: x**2./2./np.pi,xlabel='l',ylabel='D')
    #if tutils.is_planck(qid1) and tutils.is_planck(qid2): pl.add(ncents,cltt,color='k',lw=3) # unblind only if both planck
    pl.add_err(ccents,res,yerr=errs,marker="o",ls="none")
    pl.add(ccents[sel],pfit(ccents),ls='--')
    pl._ax.set_xlim(flmin,flmax)
    pl._ax.set_ylim(0.1,1e8)
    pl.hline(y=0)
    pl.done("%sres_%s_%s.png" % (spath,qid1,qid2),verbose=False)

    


sys.exit()


def physical_fit(ells):
    """
    We use a physically motivated fitting function.
    This will be 
    1. cibc with amplitude and index free
    2. cibp with amplitude and index free
    3. 
    """
    pass

region = 'deep56'

mask = enmap.read_map("/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_%s/tilec_mask.fits" % (region))
shape,wcs = mask.shape,mask.wcs
modlmap = mask.modlmap()
bin_edges = np.arange(80,8000,40)
binner = stats.bin2D(modlmap,bin_edges)
arrays = "d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08".split(',')
narrays = len(arrays)
for qind1 in range(narrays):
    for qind2 in range(qind1,narrays):
        qid1 = arrays[qind1]
        qid2 = arrays[qind2]
        fname = "/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_%s/tilec_hybrid_covariance_%s_%s.npy" % (region,qid1,qid2)
        cov = enmap.enmap(np.load(fname),wcs)
        cents,p1d = binner.bin(cov)
        pl = io.Plotter(xyscale='linlog',scalefn = lambda x: x**2./2./np.pi,xlabel='l',ylabel='D')
        pl.add(cents,p1d)
        pl._ax.set_ylim(1,1e4)
        pl.done("pow_%s_%s.png" % (qid1,qid2))












sys.exit()


def plaw(ells,a1,a2,a3,e1,e2,e3,b,ellp):
    return a1*(ells/ellp)**e1 + a2*(ells/ellp)**e2 + a3*(ells/ellp)**e3 + b

def edges_from_centers(ls):
    w = np.diff(ls)
    assert np.all(np.isclose(w,w[0]))
    w = w[0]
    edges = ls-w/2.
    return np.append(edges,ls[-1]+w/2.)


def get_act_freq(array):
    f1 = int(array.split('_')[1][1:])
    assert f1 in [90,150,220]
    return float(f1)

def get_planck_freq(array):
    f1 = int(array)
    assert f1 in [30,44,70,100,143,217,353,545,857]
    return float(f1)

def is_hfi(array):
    if int(array) in [30,44,70]:
        return False
    elif int(array) in [100,143,217,353,545,857]:
        return True
    else:
        return ValueError    


def load_spec(patch,a1,a2):
    def _load_spec(a1,a2):
        try:
            aseason1,aarray1 = a1
            str1 = "%s_%s" % (aseason1,aarray1)
        except:
            pfreq1 = a1
            str1 = "%s" % pfreq1
        try:
            aseason2,aarray2 = a2
            str2 = "%s_%s" % (aseason2,aarray2)
        except:
            pfreq2 = a2
            str2 = "%s" % pfreq2
        fname = "/scratch/r/rbond/msyriac/data/depot/actsims/spectra/spec%s_%s_%s_%s.txt" % (fftstr,patch,str1,str2)
        ls,Cls = np.loadtxt(fname,unpack=True)
        return ls,Cls
    try:
        return _load_spec(a1,a2)
    except:
        return _load_spec(a2,a1)

pfreqs = ['030','044','070','100','143','217','353','545']
fftstr = "_fft"

class Spec(object):
    def __init__(self,loc='./data/spectra/'):


        # with open(loc+'Planck_Planck.pickle', 'rb') as handle:
        #     self.pdict = pickle.load(handle)
        # with open(loc+'ACT_ACT.pickle', 'rb') as handle:
        #     self.adict = pickle.load(handle)            
        # with open(loc+'ACT_planck.pickle', 'rb') as handle:
        #     self.apdict = pickle.load(handle)

        self.pdict = {}
        self.apdict = {}
        self.adict = {}
        for patch in ['deep56','boss']:
            for i in range(len(pfreqs)):
                for j in range(i,len(pfreqs)):
                    array1 = pfreqs[i]
                    array2 = pfreqs[j]
                    self.pdict[patch,array1,array2] = load_spec(patch,array1,array2)

        acts = {'boss':{'s15':['pa1_f150','pa2_f150','pa3_f090','pa3_f150']},'deep56':{'s14':['pa1_f150','pa2_f150'],'s15':['pa1_f150','pa2_f150','pa3_f090','pa3_f150']}}

        for patch in ['deep56','boss']:
            combs = []
            for season in acts[patch].keys():
                for array in acts[patch][season]:
                    combs.append((season,array))
            ncombs = len(combs)
            for i in range(ncombs):
                season1,array1 = combs[i]
                for k in range(len(pfreqs)):
                    self.apdict[patch,pfreqs[k],season1,array1] = load_spec(patch,pfreqs[k],(season1,array1))
                for j in range(i,ncombs):
                    season2,array2 = combs[j]
                    self.adict[patch,season1,array1,season2,array2] = load_spec(patch,(season1,array1),(season2,array2))



        croot = "data/cosmo2017_10K_acc3"
        self.theory = cosmology.loadTheorySpectraFromCAMB(croot,get_dimensionless=False)

        import pandas as pd
        self.adf = pd.read_csv("./data/spectra/wnoise.csv")
        # 545 and 857 are wrong
        self.planck_rms = {'030':195,'044':226,'070':199,'100':77,'143':33,'217':47,'353':153,'545':1000,'857':1000}

    def act_rms(self,season,patch,array,freq):
        return self.adf[(self.adf['season']==season) & (self.adf['patch']==patch) & (self.adf['array']==array) & (self.adf['freq']==freq)]['wnoise'].item()
    
    def get_spec(self,patch,dm1,dm2,season1=None,array1=None,
                 season2=None,array2=None):

        print(patch,dm1,dm2,season1,array1,
              season2,array2)        
        if array2 is None:
            assert season2 is None
            season2 = season1
            array2 = array1
        if dm1=='act' and dm2=='act':
            mdict = self.adict
            try:
                r = mdict[patch,season1,array1,season2,array2]
            except:
                r = mdict[patch,season2,array2,season1,array1]
            ellmin = 2000
            ellmax = 8000
            f1 = get_act_freq(array1)
            f2 = get_act_freq(array2)
            ellp = 4000
        elif dm1=='act' and dm2=='planck':
            mdict = self.apdict
            try:
                r = mdict[patch,array1,season2,array2]
                f1 = get_planck_freq(array1)
                f2 = get_act_freq(array2)
                try:
                    assert float(array1)>90, "ACT cross with LFI not allowed under blinding"
                except:
                    return [None]*5
            except:
                r = mdict[patch,array2,season1,array1]
                f1 = get_act_freq(array1)
                f2 = get_planck_freq(array2)
                try:
                    assert float(array2)>90, "ACT cross with LFI not allowed under blinding"
                except:
                    return [None]*5
            ellmin = 1000
            ellmax = 3000
            ellp = 2500
        elif dm1=='planck' and dm2=='planck':
            mdict = self.pdict
            try:
                r = mdict[patch,array1,array2]
            except:
                r = mdict[patch,array2,array1]
            both_hfi = (is_hfi(array1) and is_hfi(array2))
            ellmin = 500 if both_hfi else 300
            ellmax = 2000 if both_hfi else 600
            ellp = 1500 if both_hfi else 500
            f1 = get_planck_freq(array1)
            f2 = get_planck_freq(array2)

        ls,cls = r
        sel = np.logical_and(ls>ellmin,ls<ellmax)
        ells = ls[sel]
        Cls = cls[sel]

        cltt = self.theory.lCl('TT',ells)
        clyy = fgs.power_tsz(ells,f1,nu2=f2)
        cltheory = cltt + clyy

        errs = self.error(ells,patch,dm1,dm2,season1=season1,array1=array1,
                          season2=season2,array2=array2)

        pfunc = lambda x,a,b,c,d,e,f,off: plaw(x,a,b,c,d,e,f,off,ellp)
        from scipy.optimize import curve_fit
        popt,pcov = curve_fit(pfunc,ells,Cls-cltheory,sigma=errs,bounds=([0,0,0,0,0,-2,0],np.inf),absolute_sigma=True)

        pfit = lambda x: pfunc(x,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6])

        sname = '_'.join([str(x) for x in [dm1,dm2,season1,array1,
                                           season2,array2]])

        Rcls = Cls-cltheory
        ls = ells
        pl = io.Plotter(xyscale='linlin',scalefn = lambda x: x**2./2./np.pi,xlabel='$\\ell$',ylabel='$D_{\\ell}$')
        pl.add_err(ls,Rcls,yerr=errs)
        fells = np.arange(ls.min(),ls.max(),1)
        pl.add(fells,pfit(fells),ls='-.')
        pl.hline()
        pl.done("fgfit_%s_fit_region.png" % sname)


        pl = io.Plotter(xyscale='linlin',scalefn = lambda x: x**2./2./np.pi,xlabel='$\\ell$',ylabel='$D_{\\ell}$')
        pl.add_err(ls,Rcls,yerr=errs)
        fells = np.arange(10,8000,1)
        pl.add(fells,pfit(fells),ls='-.')
        pl.hline()
        pl.done("fgfit_%s_full_region.png"  % sname)
        
                
        return ells,Cls,Rcls,errs,pfit

    def cl_theory(self,ells,nu1,nu2):
        return self.theory.lCl('TT',ells) + \
            fgs.power_cibp(ells,nu1,nu2,A_cibp=None,al_cib=None,Td=None) + \
            fgs.power_tsz(ells,nu1,nu2) + \
            fgs.power_cibc(ells,nu1,nu2,A_cibc=None,n_cib=None,al_cib=None,Td=None) + \
            fgs.power_radps(ells,nu1,nu2,A_ps=None,al_ps=None) + \
            fgs.power_ksz_reion(ells,A_rksz=1,fill_type="extrapolate") + fgs.power_ksz_late(ells,A_lksz=1,fill_type="extrapolate")

    def cl_noise(self,ells,patch,dm,season=None,array=None):
        # beam
        if dm=='planck':
            import soapack.interfaces as sints
            dmodel = sints.PlanckHybrid()
            lbeam = dmodel.get_beam(ells,array=array)
            lknee = 0 ; alpha = 1
            rms = self.planck_rms[array]
        elif dm=='act':
            afreq = get_act_freq(array)
            fwhm = 1.4 * 150./afreq
            lbeam = maps.gauss_beam(ells,fwhm)
            lknee = 2500 ; alpha = -4
            a,f = array.split('_')
            rms = self.act_rms(season,patch,a,f)

        from tilec.covtools import rednoise
        return rednoise(ells,rms,lknee=lknee,alpha=alpha) / lbeam**2.
    
    def error(self,ls,patch,dm1,dm2,season1=None,array1=None,
                 season2=None,array2=None):

        def _get_freq(dm,array):
            if dm=='planck': return get_planck_freq(array)
            elif dm=='act': return get_act_freq(array)
            
        nu1 = _get_freq(dm1,array1)
        nu2 = _get_freq(dm2,array2)
        
        edges = edges_from_centers(ls)
        ells = np.arange(edges[0]-np.diff(ls)[0],edges[-1]+np.diff(ls)[0],1)
        LF = cosmology.LensForecast()
        LF.loadGenericCls("11",ells,self.cl_theory(ells,nu1,nu1),ellsNls=ells,Nls=self.cl_noise(ells,patch=patch,dm=dm1,season=season1,array=array1))        
        LF.loadGenericCls("22",ells,self.cl_theory(ells,nu2,nu2),ellsNls=ells,Nls=self.cl_noise(ells,patch=patch,dm=dm2,season=season2,array=array2))        
        LF.loadGenericCls("12",ells,self.cl_theory(ells,nu1,nu2))     # Ignore 90-150 correlation
        
        if season1==season2 and array1==array2:
            spec = '11'
        else:
            spec = '12'
        if patch=='deep56':
            fsky = 500./41252.
        if patch=='boss':
            fsky = 1700/41252.
        _,errs = LF.sn(edges,fsky,spec)
        return errs
        
    
s = Spec()
patches = ['deep56','boss']
aarrays = {'deep56':['d56_%s' % str(x).zfill(2) for x in range(1,7) ], 'boss':['boss_%s' % str(x).zfill(2) for x in range(1,4) ]}
planck_arrays = ['p0%d' % x for x in range(1,9)]
dmap = {'act_mr3':'act','planck_hybrid':'planck'}

for patch in patches:
    act_arrays = aarrays[patch]
    allarrs = act_arrays+planck_arrays
    for i in range(len(allarrs)):
        for j in range(i,len(allarrs)):
            a1 = allarrs[i]
            a2 = allarrs[j]
            dm1 = dmap[sints.arrays(a1,'data_model')]
            dm2 = dmap[sints.arrays(a2,'data_model')]
            season1 = sints.arrays(a1,'season')
            array1 = sints.arrays(a1,'array')+"_"+sints.arrays(a1,'freq')
            if dm1=='planck': array1 = array1.split('_')[1]
            season2 = sints.arrays(a2,'season')
            array2 = sints.arrays(a2,'array')+"_"+sints.arrays(a2,'freq')
            if dm2=='planck': array2 = array2.split('_')[1]
            ls,Cls,Rcls,errs,pfit = s.get_spec(patch,dm1,dm2,season1=season1,array1=array1,season2=season2,array2=array2)
sys.exit()

# ls,Cls,Rcls,errs,pfit = s.get_spec('deep56','act','act',season1='s15',array1='pa2_f150')
# ls,Cls,Rcls,errs,pfit = s.get_spec('deep56','act','act',season1='s15',array1='pa3_f150')
# ls,Cls,Rcls,errs,pfit = s.get_spec('boss','act','act',season1='s15',array1='pa2_f150')
# ls,Cls,Rcls,errs,pfit = s.get_spec('boss','act','act',season1='s15',array1='pa3_f090')
# ls,Cls,Rcls,errs,pfit = s.get_spec('deep56','act','act',season1='s15',array1='pa3_f090')
# ls,Cls,Rcls,errs,pfit = s.get_spec('deep56','planck','planck',array1='100')
# ls,Cls,Rcls,errs,pfit = s.get_spec('deep56','planck','planck',array1='030')
# ls,Cls,Rcls,errs,pfit = s.get_spec('deep56','planck','planck',array1='030',array2='100')
# ls,Cls,Rcls,errs,pfit = s.get_spec('deep56','planck','planck',array1='353')
# ls,Cls,Rcls,errs,pfit = s.get_spec('deep56','planck','planck',array1='545')
# ls,Cls,Rcls,errs,pfit = s.get_spec('deep56','planck','planck',array1='217')
# ls,Cls,Rcls,errs,pfit = s.get_spec('deep56','planck','planck',array1='143')
# ls,Cls,Rcls,errs,pfit = s.get_spec('boss','planck','planck',array1='100',array2='143')
# ls,Cls,Rcls,errs,pfit = s.get_spec('boss','act','planck',season1='s15',array1='pa3_f090',array2='143')
# ls,Cls,Rcls,errs,pfit = s.get_spec('deep56','act','act',season1='s15',array1='pa3_f090',season2='s15',array2='pa3_f150')
# ls,Cls,Rcls,errs,pfit = s.get_spec('deep56','act','planck',season1='s15',array1='pa2_f150',array2='143')
# ls,Cls,Rcls,errs,pfit = s.get_spec('boss','act','planck',season1='s15',array1='pa2_f150',array2='143')
# ls,Cls,Rcls,errs,pfit = s.get_spec('deep56','act','planck',season1='s15',array1='pa3_f090',array2='100')

