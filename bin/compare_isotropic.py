from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from tilec import utils as tutils

solutions = ['tSZ','CMB','CMB-tSZ','CMB-CIB','tSZ-CMB','tSZ-CIB']
combs = ['joint','act','planck']
regions = ['boss','deep56']
name_map = {'CMB':'cmb','tSZ':'comptony','CIB':'cib'}

root = "/scratch/r/rbond/msyriac/data/depot/tilec/"

components = {}
input_names = []
for solution in solutions:
    components[solution] = solution.split('-')
    input_names.append( components[solution][0] )
input_names = sorted(list(set(input_names)))
print(components)
print(input_names)



for region in regions:


    for comb in combs:
        version = "map_$VERSION_%s" % (comb)
        savedir = tutils.get_save_path(version,region)

        for solution in solutions:

            pl = io.Plotter(xlabel='l',ylabel='r',xyscale='loglin')


            comps = "tilec_single_tile_"+region+"_"
            comps = comps + name_map[components[solution][0]]+"_"
            if len(components[solution])>1: comps = comps + "deprojects_"+ '_'.join([name_map[x] for x in components[solution][1:]]) + "_"
            comps = comps + version    

            fname = "%s%s.fits" % (savedir,comps)

            cs = []
            for isotype in ['iso_v1.0.0_rc','v1.0.0_rc']:


                lname = fname.replace('$VERSION',isotype)
                print(lname)
            

                imap = enmap.read_map(lname)


                kmask = maps.mask_kspace(imap.shape,imap.wcs, lxcut = 40, lycut = 40)
                k = enmap.fft(imap,normalize='phys') #* kmask
                p = (k*k.conj()).real
                modlmap = imap.modlmap()

                if '_act_' in fname:
                    bin_edges = np.arange(500,8000,20)
                else:
                    bin_edges = np.arange(20,8000,20)

                binner = stats.bin2D(modlmap,bin_edges)
                cents,c1d = binner.bin(p)
                cs.append(c1d.copy())


            r = (cs[0]-cs[1])/cs[1]
                

            pl.add(cents,r)
            pl.hline(y=0)
            pl.done("isocomp_%s_%s_%s.png" % (region,comb,solution) )
