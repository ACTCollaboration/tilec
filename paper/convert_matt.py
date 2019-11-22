from __future__ import print_function
from orphics import maps,io,cosmology,catalogs
from pixell import enmap
import numpy as np
import os,sys

ifile = "paper/E-D56Clusters.fits"

#catalogs.convert_hilton_catalog_to_enplot_annotate_file('public_clusters.csv',ifile,radius=15,width=3,color='red')
catalogs.convert_hilton_catalog_to_enplot_annotate_file('paper/test_public_clusters.csv',ifile,radius=15,width=3,color='red')

