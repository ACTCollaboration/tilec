"""

We will simulate a part-sky lensed Q/U map at 2 arcmin resolution.

In order to make a tiling lensing reconstruction, one has to:
1. divide the sky into overlapping tiles
2. extract each tile and filter it (e.g. constant covariance wiener filter based on ILC empirical covmat)
3. calculate the normalization for that particular filter
4. reconstruct QE
5. apply the normalization
6. simulate the above and subtract a meanfield
7. insert the reconstruction back into the full map with a cross-fading window

"""


from __future__ import print_function
from orphics import maps,io,cosmology,stats,lensing
from pixell import enmap,lensing as enlensing,powspec,utils
import numpy as np
import os,sys
from enlib import bench
from soapack import interfaces as sints
from orphics import io,maps
from orphics import mpi
comm = mpi.MPI.COMM_WORLD
