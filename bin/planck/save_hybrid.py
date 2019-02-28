import os,sys,glob
from soapack import interfaces as sints

dm = sints.PlanckHybrid()
for array in dm.arrays:
    for splitnum in range(2):
        mfname = os.path.basename(dm.get_split_fname(None,None,array,splitnum,srcfree=False))
        ifname = os.path.basename(dm.get_split_ivar_fname(None,None,array,splitnum))
        print(mfname)
        print(ifname)
