from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
from orphics import maps,io,cosmology,catalogs,stats
from pixell import enmap,reproject
import numpy as np
import os,sys
from soapack import interfaces as sints
import tilec.fg as tfg
import tilec.utils as tutils

depdict = {'cmb':[None,'tsz','cib'],'tsz':[None,'cmb','cib']}

#tdirs = ['/scratch/r/rbond/msyriac/data/depot/tilec/','/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919/']
#versions = ['v1.1.0','v1.0.0_rc']
#regions = ['boss','deep56']
#vmin = -0.2 ; vmax = 0.2

tdirs = ['/scratch/r/rbond/msyriac/data/depot/tilec/','/scratch/r/rbond/msyriac/data/depot/tilec/v1.1.0_20191127/']
versions = ['v1.1.1','v1.1.0']
regions = ['boss','deep56']
vmin = -0.01 ; vmax = 0.01

bin_edges = np.arange(20,6000,20)

for region in regions:

    mask = sints.get_act_mr3_crosslinked_mask(region)
    modlmap = mask.modlmap()
    lmap = mask.lmap()
    binner = stats.bin2D(modlmap,bin_edges)
    def pow(x,y=None):
        k = enmap.fft(x,normalize='phys')
        ky = enmap.fft(y,normalize='phys') if y is not None else k
        p = (k*ky.conj()).real
        cents,p1d = binner.bin(p)
        return p,cents,p1d

    for component in ['cmb','tsz']:
        deprojs = depdict[component]
        for deproj in deprojs:
            ps = []
            for tdir,version in zip(tdirs,versions):
                csfile = tutils.get_generic_fname(tdir,region,component,deproject=deproj,data_comb='joint',version=version)
                imap = enmap.read_map(csfile)
                _,cents,p1d = pow(imap)
                ps.append(p1d.copy())
            p1 = ps[0]
            p2 = ps[1]
            r = (p1-p2)/p2

            pl = io.Plotter('Dell')
            pl.add(cents,p1,label=versions[0],lw=1)
            pl.add(cents,p2,label=versions[1],lw=1)
            pl.done(f"vcomp_{region}_{component}_{str(deproj)}.png")

            pl = io.Plotter('rCell')
            pl.add(cents,r,lw=1)
            pl.hline(y=0)
            pl._ax.set_ylim(vmin,vmax)
            pl.done(f"vcompdiff_{region}_{component}_{str(deproj)}.png")
            
