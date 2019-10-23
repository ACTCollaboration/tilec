import os,sys
from pixell import enmap
from orphics import io

tdir = '/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919'
    

def get_tsz_map():
    pass

for region in ['deep56','boss']:
    for dcomb in ['joint','act','planck']:
        for solution in ['cmb','tsz']:
            for deproj in [None,'cib'] + [{'cmb':'tsz','tsz':'cmb'}[solution]]:
                fname = tutils.get_generic_fname(tdir,region,solution,deproj,dcomb)
                imap = enmap.downgrade(enmap.read_map(fname),4)
                bname = 'plot_' + os.path.basename(fname).replace('.fits','')
                io.hplot(imap,bname,grid=True,color={'cmb':'planck','tsz':'gray'}[solution])
                #print(os.path.basename(fname))
