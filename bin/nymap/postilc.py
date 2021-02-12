from __future__ import print_function
from pixell import enmap
from orphics import io
import numpy as np
import os,sys
import utils as cutils
import glob

version = 'test90'

opath = f'{cutils.opath}/{version}/'
dpath = opath

def get_map(string):
    cnames = sorted(glob.glob(f'{opath}{string}*.fits'))
    cmap = 0
    for cname in cnames:
        print(cname)
        cmap = cmap + enmap.read_map(cname)
    return cmap


cmap = get_map('dcmap')
enmap.write_map(f'{dpath}cmap.fits',cmap)
ymap = get_map('dymap')
enmap.write_map(f'{dpath}ymap.fits',ymap)

io.hplot(cmap,f'{dpath}cmap',mask=0,grid=True,ticks=10,downgrade=2)
io.hplot(ymap,f'{dpath}ymap',mask=0,grid=True,ticks=10,color='gray',downgrade=2)


