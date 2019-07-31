from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap
import numpy as np
import os,sys
from soapack import interfaces as sints

mask = sints.get_act_mr3_crosslinked_mask("boss",version='padded_v1')
shape,wcs = mask.shape,mask.wcs
print(np.rad2deg(enmap.pix2sky(shape,wcs,(shape[-2],shape[-1]))))
sys.exit()
io.hplot(mask,"mask.png")
print(mask.shape)
    
dm = sints.PlanckHybrid(region=mask)
ivars = dm.get_splits_ivar(['100'],ncomp=1)
splits = dm.get_splits(['100'],ncomp=1)
# print(ivars.shape)
# print(ivars[ivars<=0].size/ivars.size*100.)
# print(ivars[ivars<=0].size)
# print(splits[ivars<=0])
# print(splits[ivars<=0].max())

# splits[ivars>0] = np.nan
# splits[ivars<=0] = 1
io.hplot(splits[0,1,0],"boss.png")
