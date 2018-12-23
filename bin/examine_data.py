from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from tilec import utils as tutils,covtools,ilc

c = tutils.Config()

for ai in range(c.narrays):
    coadd = c.load_coadd_real(ai)
    io.hplot(coadd*c.mask,"fcoadd_%d" % ai)

