from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys

nsplits = 4
shape,wcs = maps.rect_geometry(width_deg = 25.,px_res_arcmin=2.0)
tsim = maps.SplitSimulator(shape,wcs,[1.5*1e-10],[148.],[20.],[nsplits],
                           lknees=[2000],alphas=[1])

fc = maps.FourierCalc(shape,wcs)
nsims = 5

modlmap = enmap.modlmap(shape,wcs)
bin_edges = np.arange(300,3000,40)
binner = stats.bin2D(modlmap,bin_edges)

s = stats.Stats()

for i in range(nsims):
    print(i)
    observed,noises = tsim.get_sim(i)

    coadd = observed[0].mean(axis=0)
    ksplits = fc.fft(observed[0])
    kcoadd = fc.fft(coadd)

    total1,c1,n1 = maps.split_calc(ksplits,ksplits,kcoadd,kcoadd,fourier_calc=fc,alt=False)
    total2,c2,n2 = maps.split_calc(ksplits,ksplits,kcoadd,kcoadd,fourier_calc=fc,alt=True)

    cents,c1d1 = binner.bin(c1)
    cents,c2d1 = binner.bin(c2)
    cents,n1d1 = binner.bin(n1)
    cents,n2d1 = binner.bin(n2)

    s.add_to_stats("c1d1",c1d1)
    s.add_to_stats("c2d1",c2d1)
    s.add_to_stats("n1d1",n1d1)
    s.add_to_stats("n2d1",n2d1)
    


    # io.plot_img(maps.ftrans(n1))
    # io.plot_img(maps.ftrans(n2))
    assert np.all(np.isclose(total1,total2))
    assert np.all(np.isclose(c1,c2))
    assert np.all(np.isclose(n1,n2))

s.get_stats()

c1d1 = s.stats['c1d1']['mean']
c2d1 = s.stats['c2d1']['mean']
n1d1 = s.stats['n1d1']['mean']
n2d1 = s.stats['n2d1']['mean']
ec1d1 = s.stats['c1d1']['err']
ec2d1 = s.stats['c2d1']['err']
en1d1 = s.stats['n1d1']['err']
en2d1 = s.stats['n2d1']['err']

pl = io.Plotter(yscale='log',scalefn=lambda x:x**2./np.pi,xlabel=io.latex.ell,ylabel=io.latex.dl)
ells = np.arange(0,3000,1)
cltt = tsim.theory.lCl('TT',ells)
pl.add(ells,cltt)
pl.add_err(cents,c1d1,yerr=ec1d1,label='regular')
pl.add_err(cents,c2d1,yerr=ec2d1,label='alt')
pl.done("testsplit.png")

pl = io.Plotter(xlabel=io.latex.ell,ylabel=io.latex.ratcl)
noise = binner.bin(tsim.ps_noises[0])[1]
pl.add(cents,(n1d1-noise)/noise,label='regular')
pl.add(cents,(n2d1-noise)/noise,label='alt')
pl.hline()
pl.done("testsplitnoise.png")
