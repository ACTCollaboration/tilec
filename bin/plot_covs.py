from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
import pickle
from szar import foregrounds as fgs
import soapack.interfaces as sints
from tilec import ilc,utils as tutils
from scipy.optimize import curve_fit

ells = np.arange(2,8000,1)


region = 'deep56'

mask = enmap.read_map("/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_%s/tilec_mask.fits" % (region))
shape,wcs = mask.shape,mask.wcs
modlmap = mask.modlmap()
bin_edges = np.arange(80,8000,40)
binner = stats.bin2D(modlmap,bin_edges)

cents = binner.centers
ctheory = ilc.CTheory(cents)

arrays = "d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08".split(',')
narrays = len(arrays)
for qind1 in range(narrays):
    for qind2 in range(qind1,narrays):
        
        qid1 = arrays[qind1]
        qid2 = arrays[qind2]

        if qid1[0]!='p' or qid2[0]!='p': continue
        fname = "/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_%s/tilec_hybrid_covariance_%s_%s.npy" % (region,qid1,qid2)
        cov = enmap.enmap(np.load(fname),wcs)
        cents,p1d = binner.bin(cov)
        pl = io.Plotter(xyscale='linlog',scalefn = lambda x: x**2./2./np.pi,xlabel='l',ylabel='D')
        pl.add(cents,p1d)

        lmin,lmax,hybrid,radial,friend,f1,fgroup,wrfit = tutils.get_specs(qid1)
        lmin,lmax,hybrid,radial,friend,f2,fgroup,wrfit = tutils.get_specs(qid2)
        beam1 = tutils.get_kbeam(qid1,cents,sanitize=False,planck_pixwin=True)
        beam2 = tutils.get_kbeam(qid2,cents,sanitize=False,planck_pixwin=True)

        fbeam1 = tutils.get_kbeam(qid1,ells,sanitize=False,planck_pixwin=True)
        fbeam2 = tutils.get_kbeam(qid2,ells,sanitize=False,planck_pixwin=True)

        #ffunc = lambda d,x,y: ctheory.get_theory_cls(f1,f2,a_cmb=x,a_gal=y)*beam1*beam2
        ffunc = lambda d,x,y,z: ctheory.get_theory_cls(f1,f2,a_cmb=x,a_gal=y,exp_gal=z)*beam1*beam2

        #res,_ = curve_fit(ffunc,cents,p1d,p0=[1,10],bounds=([0,0],[2,100]))
        res,_ = curve_fit(ffunc,cents,p1d,p0=[1,0.8,-0.7],bounds=([0,0,-2],[2,100,0]))
        #fcmb,fgal = res
        fcmb,fgal,egal = res
        #print(fcmb,fgal)
        print(fcmb,fgal,egal)

        ctheory2 = ilc.CTheory(ells)
        
        #c2 = ctheory2.get_theory_cls(f1,f2,a_cmb=fcmb,a_gal=fgal)*fbeam1*fbeam2
        c2 = ctheory2.get_theory_cls(f1,f2,a_cmb=fcmb,a_gal=fgal,exp_gal=egal)*fbeam1*fbeam2
        pl.add(ells,c2,ls="--")
        c3 = ctheory2.get_theory_cls(f1,f2,a_cmb=0,a_gal=0.8,exp_gal=-0.7)*fbeam1*fbeam2
        print(f1,f2,c3[5])
        pl.add(ells,c3,ls=":")

        pl._ax.set_ylim(1,4e7)
        pl._ax.set_xlim(0,600)
        pl.done("pow_%s_%s.png" % (qid1,qid2))
