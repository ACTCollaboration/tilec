from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from tilec import utils as tutils,ilc
from scipy.optimize import curve_fit

version = 'v1.0.0_rc_deep56'
#qids = ['d56_0%d' % x for x in range(1,7)] + ['p0%d' % x for x in range(4,9)]
qids = ['p0%d' % x for x in range(4,9)]
root = os.environ['WORK'] + '/data/depot/tilec/' + version + '/'
mask = enmap.read_map(root + 'tilec_mask.fits')
shape,wcs = mask.shape,mask.wcs
modlmap = mask.modlmap()

bin_edges = np.append(np.arange(500,3000,200) ,[3000,4000,5000,5800,7000,8000])
binner = stats.bin2D(modlmap,bin_edges)

def efunc(ells,a1,e1,w):
    return w + a1*(ells/1000)**e1

narrays = len(qids)
aspecs = tutils.ASpecs().get_specs

for i in range(narrays):
    for j in range(i,narrays):
        qid1 = qids[i]
        qid2 = qids[j]

        kbeam1 = 1
        kbeam2 = 1

        fname = root + 'tilec_hybrid_covariance_%s_%s.npy' % (qid1,qid2)
        p2d = enmap.enmap(np.load(fname))

        cents,p1d = binner.bin(p2d/kbeam1/kbeam2)

        pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='C')
        if tutils.is_planck(qid1) or tutils.is_planck(qid2):
            pl.add(cents[cents<5800],p1d[cents<5800])
        else:
            pl.add(cents,p1d)


        if tutils.is_planck(qid1) and tutils.is_planck(qid2):
            sel = np.where(np.logical_and(cents>500,cents<4000))
            pf1 = p1d[sel][-1] if i==j else 0

            lmin,lmax,hybrid,radial,friend,f1,fgroup,wrfit = aspecs(qid1)
            lmin,lmax,hybrid,radial,friend,f2,fgroup,wrfit = aspecs(qid2)

            lbeam1 = tutils.get_kbeam(qid1,cents,sanitize=True,planck_pixwin=True)
            lbeam2 = tutils.get_kbeam(qid2,cents,sanitize=True,planck_pixwin=True)
            
            
            ctheory = ilc.CTheory(cents)
            fp1d = ctheory.get_theory_cls(f1,f2,a_gal=0.8) *lbeam1 * lbeam2 + pf1
            pl.add(cents,fp1d,ls="--")


            
        pl.done("bpower_%s_%s.png" % (qid1,qid2))
