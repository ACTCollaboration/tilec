from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
import utils as cutils
from soapack import interfaces as sints
from actsims import noise as anoise

version = 'test90'
dfact = 1


qids = cutils.qids
opath = f'{cutils.opath}/{version}/'


try: os.makedirs(opath)
except:
    pass


dm = sints.DR5()


mask = dm.get_binary_apodized_mask(qids[0],galcut=90)
enmap.write_map(f'{opath}mask.fits',cutils.down(mask,dfact))
sys.exit()

for qid in qids:
    
    imap = dm.get_splits(qid,ncomp=1,warn=False)[:,0,...]
    ivar = dm.get_ivars(qid,ncomp=1,warn=False)[:,0,...]


    nsplits = imap.shape[0]

    if nsplits>1:
        imap,ivar = anoise.get_coadd(imap,ivar,axis=0)
    else:
        imap = imap[0]
        ivar = ivar[0]


    enmap.write_map(f'{opath}{qid}_map.fits',cutils.down(imap,dfact))
    enmap.write_map(f'{opath}{qid}_ivar.fits',cutils.down(ivar,dfact,op=np.sum))
    print(qid)
