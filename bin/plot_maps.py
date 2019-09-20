import os,sys
from pixell import enmap
from orphics import io

tdir = '/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919'


def _get_generic_fname(tdir,region,solution,deproject=None,data_comb='joint',version=None,sim_index=None,beam=False,noise=False,cross_noise=False,mask=False):
    """
    Implements the naming convention in the release directory.
    """

    assert sum([int(x) for x in [beam,noise,cross_noise,mask]]) <= 1
    data_comb = data_comb.lower().strip()
    solution = solution.lower().strip()

    if sim_index is not None:
        data_comb = {'joint':'joint','act':'act_only','act_only':'act_only','planck':'planck_only','planck_only':'planck_only'}[data_comb]
        version = "test_sim_baseline_00_%s" % str(sim_index).zfill(4)
    else:
        data_comb = {'joint':'joint','act':'act','act_only':'act','planck':'planck','planck_only':'planck'}[data_comb]
        if version is None: version = "v1.0.0_rc"
    
    solution = {'cmb':'cmb','ksz':'cmb','tsz':'comptony','comptony':'comptony'}[solution]
    if deproject is None:
        dstr = '' 
    else:
        deproject = deproject.lower().strip()
        deproject = {'cmb':'cmb','ksz':'cmb','tsz':'comptony','comptony':'comptony','cib':'cib','dust':'cib'}[deproject]
        dstr = '_deprojects_%s' % deproject
    if sim_index is None:
        if mask: suff = "tilec_mask.fits"
        else: suff = "tilec_single_tile_%s_%s%s_map_%s_%s.fits" % (region,solution,dstr,version,data_comb)
        rval = tdir + "/map_%s_%s_%s/%s" % (version,data_comb,region,suff)
    else:
        if mask: suff = "tilec_mask.fits"
        else: suff = "tilec_single_tile_%s_%s%s_map_%s_%s.fits" % (region,solution,dstr,data_comb,version)
        rval = tdir + "/map_%s_%s_%s/%s" % (data_comb,version,region,suff)

    if beam: 
        rval = rval[:-5] + "_beam.txt"
    elif noise:
        rval = rval[:-5] + "_noise.fits"
    elif cross_noise:
        rval = rval[:-5] + "_cross_noise.fits"
    return rval
    

def get_tsz_map():
    pass

for region in ['deep56','boss']:
    for dcomb in ['joint','act','planck']:
        for solution in ['cmb','tsz']:
            for deproj in [None,'cib'] + [{'cmb':'tsz','tsz':'cmb'}[solution]]:
                fname = _get_generic_fname(tdir,region,solution,deproj,dcomb)
                imap = enmap.downgrade(enmap.read_map(fname),4)
                bname = 'plot_' + os.path.basename(fname).replace('.fits','')
                io.hplot(imap,bname,grid=True,color={'cmb':'planck','tsz':'gray'}[solution])
                #print(os.path.basename(fname))
