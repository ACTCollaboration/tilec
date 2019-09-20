from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from orphics import maps,io,cosmology,stats
from pixell import enmap,curvedsky as cs
import numpy as np
import os,sys
from soapack import interfaces as sints
from tilec import utils as tutils
import healpy as hp


dm = sints.PlanckHybrid()
omap = dm.get_splits(season=None,patch=None,arrays=['353'],ncomp=1,srcfree=False)[0,0,0][6000:,:] #[0,:,:,:,:]
print(omap.shape)


gmap = omap.copy()



for region,mul in zip(['boss','deep56'],[1.3,1]):
    print(region)
    mask = sints.get_act_mr3_crosslinked_mask(region)
    omap.insert(mask*gmap.max()/100*mul,omap,op=np.ndarray.__iadd__)

io.hplot(enmap.downgrade(omap,8),'fig_footprint',grid=True,ticks=10,font_size=60)
# plt.clf()
# hp.mollview(tmap,cbar = False)
# hp.graticule()
# plt.savefig("hmap.png")
