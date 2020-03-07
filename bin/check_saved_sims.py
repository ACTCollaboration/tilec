from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap
import numpy as np
import os,sys


#for region in ['boss','deep56']:
for region in ['deep56']:

    c1 = 0
    c2 = 0
    for isimid in range(160,500):


        simid = str(isimid).zfill(4)
        fname1 = f"/scratch/r/rbond/msyriac/data/depot/tilec/map_joint_v1.1.0_sim_baseline_00_{simid}_{region}/tilec_single_tile_{region}_cmb_map_joint_v1.1.0_sim_baseline_00_{simid}.fits"
        fname2 = f"/scratch/r/rbond/msyriac/data/depot/tilec/map_joint_v1.1.0_sim_baseline_00_{simid}_{region}/tilec_single_tile_{region}_cmb_deprojects_comptony_map_joint_v1.1.0_sim_baseline_00_{simid}.fits"

        if not(os.path.exists(fname1)): 
            iid = isimid - (isimid//10)*10
            if iid<=5: print(region,simid,"nodeproj")
            c1 = c1 + 1

        if not(os.path.exists(fname2)): 
            # print(region,simid,"deproj")
            c2 = c2 + 1

    print(region,c1,c2)

