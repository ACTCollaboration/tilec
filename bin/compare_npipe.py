#%%
from pixell import enmap,curvedsky as cs, utils as u
import numpy as np
from orphics import maps,io
import os,sys
# %%

def load_map(version):
    box = np.asarray([[-10,-10],[10,10]]) * u.degree
    imaps = []
    ivars = []
    for i in range(2):
        if version=='v18':
            fname = f"/scratch/r/rbond/msyriac/data/planck/data/hybrid/planck_hybrid_545_2way_{i}_map.fits"
            ifname = fname = f"/scratch/r/rbond/msyriac/data/planck/data/hybrid/planck_hybrid_545_2way_{i}_ivar.fits"
        elif version=='v20':
            fname = f"/home/r/rbond/sigurdkn/project/actpol/planck/npipe/car_equ/planck_npipe_545_split{i+1}_map.fits"
            ifname = f"/home/r/rbond/sigurdkn/project/actpol/planck/npipe/car_equ/planck_npipe_545_split{i+1}_ivar.fits"

        p18 = enmap.downgrade(enmap.read_map(fname,box=box,sel=np.s_[0,...]),4)
        ip18 = enmap.downgrade(enmap.read_map(ifname,box=box,sel=np.s_[0,...]),4,op=np.sum)
        print(p18.shape,ip18.shape)
        
    return p18
#%%
p18 = load_map('v20')
io.hplot(p18,colorbar=True,range=40000)
# %%
p20 = load_map()
io.hplot(p20,colorbar=True,range=40000)

# %%
taper,_ = maps.get_taper_deg(p18.shape,p18.wcs)
bin_edges = np.arange(100,3000,80)
cents,c18 = maps.binned_power(p18,bin_edges=bin_edges,mask=taper)
cents,c20 = maps.binned_power(p20,bin_edges=bin_edges,mask=taper)
# %%
pl = io.Plotter('Dell')
pl.add(cents,c18,label='Planck 2018')
pl.add(cents,c20,label='Planck 2020')
pl.done()
# %%
