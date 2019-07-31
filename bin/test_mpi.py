from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from orphics import maps,io,cosmology,mpi,stats
from pixell import enmap,curvedsky,utils
import numpy as np
import os,sys,shutil
from actsims.util import seed_tracker
from soapack import interfaces as sints
from enlib import bench
from tilec import pipeline,utils as tutils
import healpy as hp
from szar import foregrounds as fgs
from datetime import datetime

nsims = 8
bins = 10

def get_set1():
    return np.random.normal(loc=2e14,scale=0.1e14,size=(bins,))

def get_set2():
    return np.random.normal(loc=1e-14,scale=0.1e-14,size=(bins,))


comm,rank,my_tasks = mpi.distribute(nsims)


results = {}

for task in my_tasks:
    
    vec = get_set1()
    try: results['1'] = results['1'] + vec
    except: results['1'] = vec.copy()


    vec = get_set2()
    try: results['2'] = results['2'] + vec
    except: results['2'] = vec.copy()


results['1'] = utils.allreduce(results['1'],comm)/nsims
results['2'] = utils.allreduce(results['2'],comm)/nsims

if rank==0:
    print(results['1'])
    print(results['2'])
