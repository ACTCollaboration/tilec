from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys


opath = "/scratch/r/rbond/msyriac/data/depot/nymap"
qids = [f'ap19_{f}' for f in ['090','150','220']] + ['p07','p08']



def down(imap,dfact,op=np.mean):
    if (dfact is not None) and dfact!=1:
        return enmap.downgrade(imap,dfact,op=op)
    else:
        return imap



def get_mlmax(qid):
    if qid in ['p07','p08']:
        return 6144
    else:
        return 24576
