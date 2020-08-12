from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from tilec import pipeline
from soapack import interfaces as sints

"""

qids are short names that identify map/arrays in soapack/soapack/data/all_arrays_dr5.csv

I have added qids corresponding to Sigurd's coadds.

"""


# Let's downgrade the maps by 4x for debugging
dfact = 4
# Choose a version name (used for the output directory)
version = 'test_needlets_ocoadd'

# Get the soapack data model
dm = sints.DR5()

# We work with just the ACT+Planck 2018 night-time coadd
qids = ['ap18_n150']

# We get the geometry corresponding to this patch
shape,wcs = sints.get_geometry('ap18_n150')

"""
Pre-defined sets of needlet specs are in 
data/needlet_bounds_{X}.txt 
data/needlet_lmaxs_{X}.txt 

where I have saved files for X = lensmode and X = szmode
The former only goes out to ell of 4000 (useful for lensing and maybe NG)
and so is less resource intensive to run. The latter goes out
to the ACT resolution limit.

The bounds file specifies the ell range to include for
various maps, specified separately for the Planck 30 - 545 GHz channels
(qids p01 to p08), 'act' maps and 'actplanck' maps. The latter is
relevant for ACT+Planck coadds. 

The lmaxs file specifies the lmaxs of the band-limited needlet
scheme I use (see Apppendix B of 1807.06208). For each of those scales,
I also specify the pixel width in arcminutes of the CAR maps using
which the pixel level operations are done.

Let's work with lensmode for this example
"""

mode = 'lensmode'

# This is only relevant for coadding, but with the current set up
# you are forced to use it. It reconvolves all maps to the same
# beam FWHM
target_fwhm = 1.5

# Define a function that provides either the shape,wcs geometry (geom=True) of the mask
# or the mask itself. We use the default apodized binary mask in soapack.
def mask_fn(qid,geom=False):
    if geom: return enmap.read_map_geometry(dm.get_binary_apodized_mask_fname(qid))
    else: return dm.get_binary_apodized_mask(qid)

# Populate a dictionary with the mask geometries
mgeos = {}
for qid in qids:
    mgeos[qid] = mask_fn(qid,geom=True)

# Save the needlet covariance maps
pipeline.make_needlet_cov(version,qids,target_fwhm,mode,shape,wcs,mask_fn=mask_fn,mask_geometries=mgeos,dfact=dfact,overwrite=True)


