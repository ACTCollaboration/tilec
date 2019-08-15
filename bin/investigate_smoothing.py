from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from tilec import covtools
from tilec.ilc import CTheory

qid = 'p01'

scov = enmap.read_map("/scratch/r/rbond/msyriac/dump/unsmoothed_debeamed_%s_%s.fits" % (qid,qid))
dncov = enmap.read_map("/scratch/r/rbond/msyriac/dump/smoothed_noise_%s_%s.fits" % (qid,qid))
beamsq = enmap.read_map("/scratch/r/rbond/msyriac/dump/beamsq_%s_%s.fits" % (qid,qid))

io.power_crop(scov,200,"inv_scov.png")
io.power_crop(dncov,200,"inv_dncov.png")
io.power_crop(beamsq,200,"inv_beamsq.png")

bin_edges = np.arange(80,3000,20)
modlmap = scov.modlmap()
binner = stats.bin2D(modlmap,bin_edges)

# fwhm = 33.
# lefts = bin_edges[:-1]
# rights = bin_edges[1:]
# cents = binner.centers
# ms = maps.gauss_beam(rights,fwhm)/maps.gauss_beam(lefts,fwhm)
# pl = io.Plotter(xyscale='linlin',xlabel='l',ylabel='Bright/Bleft')
# pl.add(cents,ms)
# pl.done("inv_brat.png")
# sys.exit()

cents,s1d = binner.bin(scov)
cents,bs1d = binner.bin(scov*beamsq)
cents,n1d = binner.bin(dncov)

pl = io.Plotter(xyscale='linlog',scalefn = lambda x: x**2./2./np.pi,xlabel='l',ylabel='D')
pl.add(cents,s1d,'unsmoothed signal')
pl.add(cents,bs1d,'unsmoothed signal * beamsq')
pl.add(cents,n1d,'smoothed noise')
pl._ax.set_ylim(1,1e5)
pl.done("inv_pow.png")

signal_bin_width = 160
signal_interp_order = 1
slmin = 80
smsig1 = covtools.signal_average(scov,bin_width=signal_bin_width,
                                kind=signal_interp_order,
                                lmin=slmin,
                                dlspace=True)  * beamsq

smsig2 = covtools.signal_average(scov*beamsq,bin_width=signal_bin_width,
                                kind=signal_interp_order,
                                lmin=slmin,
                                dlspace=True)


mtheory = CTheory(modlmap)
f1 = 30 ; f2 = 30
ptheory = mtheory.get_theory_cls(f1,f2,a_gal=0)
rcov = scov / ptheory
rcov[~np.isfinite(rcov)] = 0
smsig3 = covtools.signal_average(rcov,bin_width=signal_bin_width,
                                 kind=signal_interp_order,
                                 lmin=slmin,
                                 dlspace=False)  * ptheory * beamsq

# smsig4 = covtools.signal_average(rcov,bin_width=signal_bin_width,
#                                  kind=signal_interp_order,
#                                  lmin=slmin,
#                                  dlspace=True)  * ptheory * beamsq


smsig5 = covtools.signal_average(rcov*beamsq,bin_width=signal_bin_width,
                                 kind=signal_interp_order,
                                 lmin=slmin,
                                 dlspace=False)  * ptheory 

smsig6 = covtools.signal_average(scov,bin_width=signal_bin_width,
                                kind=0,
                                lmin=slmin,
                                dlspace=True)  * beamsq


cents,s1d1 = binner.bin(smsig1)
cents,s1d2 = binner.bin(smsig2)
cents,s1d3 = binner.bin(smsig3)
#cents,s1d4 = binner.bin(smsig4)
cents,s1d5 = binner.bin(smsig5)
cents,s1d6 = binner.bin(smsig6)

pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='C') # ,scalefn = lambda x: x**2./2./np.pi
pl.add(cents,s1d1,'smooth(signal)*beamsq')
pl.add(cents,s1d2,'smooth(signal*beamsq)')
pl.add(cents,s1d3,'smooth(signal/theory)*beamsq*theory')
#pl.add(cents,s1d4,'smooth(signal/theory)*beamsq*theory in D_ell')
pl.add(cents,s1d5,'smooth(signal/theory*beamsq)*theory')
pl.add(cents,s1d6,'smooth(signal)*beamsq 0th')
pl.add(cents,n1d,'smoothed noise')
#pl._ax.set_ylim(1,1e5)
pl.legend(loc='upper right')
pl.done("inv_pow2.png")


cents,b1d = binner.bin(beamsq)
pl = io.Plotter(xyscale='loglog',xlabel='l',ylabel='B')
pl.add(cents,b1d)
pl.hline(y=1)
pl.done("inv_beam.png")
