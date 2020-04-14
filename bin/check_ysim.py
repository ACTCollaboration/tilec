from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from tilec import utils as tutils

region = 'deep56'
#region = 'boss'
solution = 'comptony'
tdir = "/scratch/r/rbond/msyriac/data/depot/tilec/v1.2.0_20200324"
dcomb = 'joint'

dfile = tutils.get_generic_fname(tdir,region,solution,deproject=None,data_comb=dcomb,version=None,sim_index=None)
dbeam = tutils.get_generic_fname(tdir,region,solution,deproject=None,data_comb=dcomb,version=None,sim_index=None,beam=True)
sfile = tutils.get_generic_fname(tdir,region,solution,deproject=None,data_comb=dcomb,version=None,sim_index=0)
sbeam = tutils.get_generic_fname(tdir,region,solution,deproject=None,data_comb=dcomb,version=None,sim_index=0,beam=True)
tfile = tutils.get_generic_fname(tdir,region,solution,deproject=None,data_comb=dcomb,version=None,sim_index=0)

cdfile = tutils.get_generic_fname(tdir,region,"cmb",deproject=None,data_comb=dcomb,version=None,sim_index=None)
cdbeam = tutils.get_generic_fname(tdir,region,"cmb",deproject=None,data_comb=dcomb,version=None,sim_index=None,beam=True)
csfile = tutils.get_generic_fname(tdir,region,'cmb',deproject=None,data_comb=dcomb,version=None,sim_index=0)
csbeam = tutils.get_generic_fname(tdir,region,'cmb',deproject=None,data_comb=dcomb,version=None,sim_index=0,beam=True)
ctfile = tutils.get_generic_fname(tdir,region,'cmb',deproject=None,data_comb=dcomb,version=None,sim_index=0)


dmap = enmap.read_map(dfile)
smap = enmap.read_map(sfile)
tmap = enmap.read_map(tfile)

cdmap = enmap.read_map(cdfile)
csmap = enmap.read_map(csfile)
ctmap = enmap.read_map(ctfile)

modlmap = dmap.modlmap()

ls,db = np.loadtxt(dbeam,unpack=True)
dbeam = maps.interp(ls,db)(modlmap)

ls,cdb = np.loadtxt(cdbeam,unpack=True)
cdbeam = maps.interp(ls,cdb)(modlmap)

ls,sb = np.loadtxt(sbeam,unpack=True)
sbeam = maps.interp(ls,sb)(modlmap)

ls,csb = np.loadtxt(csbeam,unpack=True)
csbeam = maps.interp(ls,csb)(modlmap)



#io.hplot(smap,"simmap")

bin_edges = np.arange(20,6000,20)
binner = stats.bin2D(modlmap,bin_edges)
p = lambda x: binner.bin((x*x.conj()).real)

dk = enmap.fft(dmap,normalize='phys')/dbeam
sk = enmap.fft(smap,normalize='phys')/sbeam
# tk = enmap.fft(tmap,normalize='phys')/sbeam

cdk = enmap.fft(cdmap,normalize='phys')/cdbeam
csk = enmap.fft(csmap,normalize='phys')/csbeam
# ctk = enmap.fft(ctmap,normalize='phys')/ctbeam

cents,d1d = p(dk)
cents,s1d = p(sk)
# cents,t1d = p(tk)

cents,cd1d = p(cdk)
cents,cs1d = p(csk)
# cents,ct1d = p(ctk)

pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='$D^{yy}_l$' ,scalefn = lambda x: x**2./2./np.pi)
#pl = io.Plotter('Dell')
pl.add(cents,d1d,label='data')
pl.add(cents,s1d,label='sim')
# pl.add(cents,t1d,label='new sim')
#pl._ax.set_ylim(1e-14,5e-10)
pl.done("dcomp.png")

pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='$D^{\\rm{CMB}}_l$' ,scalefn = lambda x: x**2./2./np.pi)
pl.add(cents[cents>5000],cd1d[cents>5000],label='data') # blinding ACT CMB data ell<5000
pl.add(cents,cs1d,label='sim')
# pl.add(cents,ct1d,label='new sim')
#pl._ax.set_ylim(1e-14,5e-10)
pl.done("cdcomp.png")


# pl = io.Plotter(xyscale='linlin',xlabel='l',ylabel='$D^{\\rm{CMB-new}}_l / D^{\\rm{CMB-old}}_l$')
# pl.add(cents,ct1d/cs1d)
# pl.hline(y=1)
# pl._ax.set_ylim(0.85,1.05)
# pl.done("cdcompdiff.png")
