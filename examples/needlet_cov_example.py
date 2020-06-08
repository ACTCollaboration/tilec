from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from tilec import pipeline
from soapack import interfaces as sints

dfact = 4
version = 'test_needlets'
#qids =  sints.get_all_dr5_qids() #['p04','p05','p06','d6','d56_05','d56_06','s18_02']
qids = ['p04','mbac_01','d6','s16_01']
#shape,wcs = sints.get_geometry('d56_05')
shape,wcs = sints.get_geometry('d6')
mode = 'lensmode'
target_fwhm = 1.5
pipeline.make_needlet_cov(version,qids,target_fwhm,mode,shape,wcs,dfact=dfact,mask_fn=mask_fn)
