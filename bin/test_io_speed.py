from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap
import numpy as np
import os,sys
import soapack.interfaces as sints
from enlib import bench

ntrials = 10
loc = "/scratch/r/rbond/msyriac/data/depot/tilec/temp/"



mask = sints.get_act_mr3_crosslinked_mask("boss")

for i in range(ntrials):
    with bench.show("write"):
        enmap.write_map(loc+"trial_%d.fits" % i,mask)


for i in range(ntrials):
    with bench.show("read"):
        mask = enmap.read_map(loc+"trial_%d.fits" % i)


for i in range(ntrials):
    os.remove(loc+"trial_%d.fits" % i)



