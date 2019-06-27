import os,sys,glob
from soapack import interfaces as sints
from pixell import enmap

"""

Planck Hybrid spec

f1: Temperature/LFI : 2015, ringhalf, 1024
f2: Temperature/HFI : 2015, halfmission, 2048
f3: Polarization/LFI : 2018, ringhalf, 1024, BPass
f4: Polarization/HFI : 2018, odd/even, 2048


"""
pr2_loc = "/scratch/r/rbond/msyriac/data/planck/data/pr2/"
pr3_loc = "/scratch/r/rbond/msyriac/data/planck/data/pr3/"

def make_hybrid(tfile,pfile):
    # Make a hybrid TQU map from specified temperature and polarization files
    T = enmap.read_map(tfile, sel=(0,))
    if ("545" in tfile) or ("857" in tfile): npol = 1
    else: npol = 3
    omap = enmap.zeros((npol,)+T.shape, T.wcs, T.dtype)
    omap[0] = T
    if npol==3: omap[1:] = enmap.read_map(pfile, sel=(slice(1,3),))
    return omap

lfis = ['030','044','070']
hfis = ['100','143','217','353','545','857']

# Input file name convention
f1 = lambda array,splitnum,mtype: "%s/LFI_SkyMap_%s_1024_R2.01_full-ringhalf-%d_%s.fits" % (pr2_loc,array,splitnum+1,mtype)
f2 = lambda array,splitnum,mtype: "%s/HFI_SkyMap_%s_2048_R2.02_halfmission-%d_%s.fits" % (pr2_loc,array,splitnum+1,mtype)
f3 = lambda array,splitnum,mtype: "%s/LFI_SkyMap_%s-BPassCorrected_1024_R3.00_full-ringhalf-%d_%s.fits" % (pr3_loc,array,splitnum+1,mtype)
def f4(array,splitnum,mtype): 
    farray = array if array!='353' else '353-psb'
    if splitnum==0: splitname = "odd" 
    elif splitnum==1: splitname = "even" 
    return "%s/HFI_SkyMap_%s_2048_R3.01_full-%sring_%s.fits" % (pr3_loc,farray,splitname,mtype)

# Get output filename convention
dm = sints.PlanckHybrid()
save_loc = dm.config['maps_path']
for array in dm.arrays:
    if array in lfis: continue # !!! REMOVE
    for splitnum in range(2):
        if array in lfis:
            mtfile = f1(array,splitnum,"map")
            mpfile = f3(array,splitnum,"map")
            itfile = f1(array,splitnum,"ivar")
            ipfile = f3(array,splitnum,"ivar")
        elif array in hfis:
            mtfile = f2(array,splitnum,"map")
            mpfile = f4(array,splitnum,"map")
            itfile = f2(array,splitnum,"ivar")
            ipfile = f4(array,splitnum,"ivar")
        else:
            raise ValueError

        # Save hybrids based on output filename convention
        mfname = os.path.basename(dm.get_split_fname(None,None,array,splitnum,srcfree=False))
        mmap = make_hybrid(mtfile,mpfile)
        enmap.write_map(save_loc+mfname,mmap)
        print("Saved ",save_loc+mfname)
        ifname = os.path.basename(dm.get_split_ivar_fname(None,None,array,splitnum))
        imap = make_hybrid(itfile,ipfile)
        enmap.write_map(save_loc+ifname,imap)
        print("Saved ",save_loc+ifname)
