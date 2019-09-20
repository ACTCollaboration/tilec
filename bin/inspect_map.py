from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
from enlib import bench
import numpy as np
import os,sys
from tilec import pipeline,utils as tutils
from soapack import interfaces as sints

"""
"""

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("version", type=str,help='Region name.')
parser.add_argument("region", type=str,help='Region name.')
parser.add_argument("solution", type=str,help='Solution.')
parser.add_argument("--lmin",     type=int,  default=80,help="lmin.")
parser.add_argument("--lmax",     type=int,  default=6000,help="lmin.")
args = parser.parse_args()

save_path = sints.dconfig['tilec']['save_path']
savedir = save_path + args.version + "_" + args.region
mask = enmap.read_map("%s/tilec_mask.fits" % savedir)
dm = sints.ACTmr3(region=mask)
fbeam = lambda x: dm.get_beam(x, "s15","deep56","pa3_f090", kind='normalized')

name_map = {'CMB':'cmb','tSZ':'comptony','CIB':'cib'}
comps = "tilec_single_tile_"+args.region+"_" + name_map[args.solution]+"_"+args.version
lmin = args.lmin
lmax = args.lmax

w2 = np.mean(mask**2.)
imap = enmap.read_map("%s/%s.fits" % (savedir,comps))
color = 'planck' if args.solution=='CMB' else 'gray'
io.hplot(imap,"map_%s_%s" % (args.solution,args.region),color='planck',grid=True)
imap1 = dm.get_coadd(season="s15",patch=args.region,array="pa3_f090",srcfree=True,ncomp=1)
imap2 = dm.get_coadd(season="s15",patch=args.region,array="pa3_f150",srcfree=True,ncomp=1)
#io.hplot(imap2,"cmb_map_%s_s15_pa3_f150" % (args.region),color="planck",grid=True)
#io.hplot(imap1,"cmb_map_%s_s15_pa3_f090" % (args.region),color="planck",grid=True)




nmap = enmap.read_map("%s/%s_noise.fits" % (savedir,comps))
ls,bells = np.loadtxt("%s/%s_beam.txt" % (savedir,comps),unpack=True)
modlmap = imap.modlmap()
kmap = enmap.fft(imap,normalize="phys")
p2d = np.real(kmap*kmap.conj())

if args.solution=='tSZ':
    lim = [-19,-15]
else:
    lim = [-5,1]

    #omap = enmap.read_map('/scratch/r/rbond/msyriac/data/tilec/omar/dataCoadd_combined_I_s14&15_deep56.fits')
    omap = enmap.read_map('/scratch/r/rbond/msyriac/data/tilec/omar/preparedMap_T_s14&15_deep56.fits')*2.726e6
    omar_mask = enmap.read_map('/scratch/r/rbond/msyriac/data/tilec/omar/mask_s14&15_deep56.fits')
    omar_w2 = np.mean(omar_mask**2.)
    # io.hplot(enmap.downgrade(omap,4),"omap")
    # io.hplot(enmap.downgrade(omask,4),"omask")
    # sys.exit()
    okmap = enmap.fft(omap,normalize='phys')
    op2d = np.real(okmap*okmap.conj())
    

io.power_crop(p2d,200,"pimg_%s_%s.png"  % (args.solution,args.region),lim=lim)
io.power_crop(nmap,200,"nimg_%s_%s.png"  % (args.solution,args.region),lim=lim)

sel = np.logical_and(modlmap>lmin,modlmap<lmax)
xs = modlmap[sel].reshape(-1)
ys = p2d[sel].reshape(-1)

pl = io.Plotter(xyscale='linlog',xlabel='l',ylabel='C')
pl._ax.scatter(xs,ys)
#pl._ax.set_ylim(lim[0],lim[1])
pl.done("pscatter_%s_%s.png" % (args.solution,args.region))

bin_edges = np.arange(80,10000,80)
binner = stats.bin2D(modlmap,bin_edges)
cents,p1d = binner.bin(p2d/w2)
cents,n1d = binner.bin(nmap*maps.interp(ls,bells)(modlmap)**2)
if args.solution=='CMB':
    #cents,omar_p1d = binner.bin(op2d*(maps.interp(ls,bells)(modlmap)**2)/omar_w2/(fbeam(modlmap)**2.))
    cents,omar_p1d = binner.bin(op2d*(maps.interp(ls,bells)(modlmap)**2)/omar_w2)
    


omask = enmap.read_map('/scratch/r/rbond/msyriac/tiling/tilec_deep56_apodized_mask.fits')
ow2 = np.mean(omask**2.)
if args.solution=='tSZ':
    imap = enmap.read_map('/scratch/r/rbond/msyriac/tiling/tilec_deep56_comptony_map_deprojects_nothing_v0.2.3_act_planck.fits')
else:
    imap = enmap.read_map('/scratch/r/rbond/msyriac/tiling/tilec_deep56_cmb_map_deprojects_nothing_v0.2.3_act_planck.fits')
kmap = enmap.fft(imap,normalize='phys')
modlmap = imap.modlmap()
binner = stats.bin2D(modlmap,bin_edges)
p2d = np.real(kmap*kmap.conj())
cents,op1d = binner.bin(p2d*maps.interp(ls,bells)(modlmap)**2/ow2)

pl = io.Plotter(xyscale='loglog',xlabel='l',ylabel='D',scalefn = lambda x: x**2./2./np.pi)
pl.add(cents,p1d,label='ILC v1 release candidate auto')
pl.add(cents,op1d,ls=':',label='ILC v0.2.3 auto')
if args.solution=='CMB':
    pl.add(cents,omar_p1d,ls='-.',label='k-space auto')
#pl.add(cents,n1d,ls='--',label='ILC v1 noise cov')
#pl._ax.set_ylim(2e1,1e4)
pl.done("p1d_%s_%s.png"  % (args.solution,args.region))


