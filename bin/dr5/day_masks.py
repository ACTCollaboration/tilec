from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys

"""
This script loads Emilie's north-daytime 
masks, trims it and saves it.
"""


for season in ['s14','s15','s16']:
    if season=='s16':
        Nx = 6000
        Ny = 1000
    else:
        Nx = 1200
        Ny = None
    imap = enmap.read_map(f"/home/r/rbond/msyriac/data/act/maps/emilie/{season}_mask_final.fits")
    omap = imap[Ny:,:-Nx]
    enmap.write_map(f"/home/r/rbond/msyriac/data/act/maps/emilie/{season}_mask_final_trimmed.fits",omap)
    #io.hplot(imap,f'daymask_{season}',grid=True,mask=0,ticks=20)
