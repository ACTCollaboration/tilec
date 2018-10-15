
def get_beams(ai,aj): # get beam fwhm for array indices ai and aj
    return darrays[arrays[ai]]['beam'],darrays[arrays[aj]]['beam']
def isplanck(aindex): # is array index ai a planck array?
    name = darrays[arrays[aindex]]['name'].lower()
    return True if ("hfi" in name) or ("lfi" in name) else False
def is90150(ai,aj): # is array index ai,aj an act 150/90 combination?
    iname = darrays[arrays[ai]]['name'].lower()
    jname = darrays[arrays[aj]]['name'].lower()
    return True if (("90" in iname) and ("150" in jname)) or (("90" in jname) and ("150" in iname)) else False
# Functions for filenames corresponding to an array index
def coaddfname(aindex): return darrays[arrays[aindex]]['froot'] + darrays[arrays[aindex]]['coadd']
def cinvvarfname(aindex): return darrays[arrays[aindex]]['froot'] + darrays[arrays[aindex]]['cinvvar']
def splitfname(aindex,split):
    try: splitstart = darrays[arrays[aindex]]['splitstart']
    except: splitstart = 0
    return darrays[arrays[aindex]]['froot'] + darrays[arrays[aindex]]['splits'] % (split + splitstart)
def sinvvarfname(aindex,split):
    try: splitstart = darrays[arrays[aindex]]['splitstart']
    except: splitstart = 0
    return darrays[arrays[aindex]]['froot'] + darrays[arrays[aindex]]['sinvvars'] % (split + splitstart)
def get_nsplits(aindex): return darrays[arrays[aindex]]['nsplits']
def xmaskfname(): return config['xmask']




def get_coadds():
    """
    Should also beam deconvolve and low pass here
    (narray,Ny,Nx)
    """
    kcoadds = []
    wins = []
    mask = enmap.read_map(xmaskfname())
    for ai in range(narrays):
        # iwin = enmap.read_map(cinvvarfname(ai))[0]
        iwin = 1.
        window = mask*iwin
        wins.append(window)
        imap = enmap.read_map(coaddfname(ai),sel=np.s_[0,:,:])
        _,_,kcoadd = fc.power2d(window*imap)
        ifwhm,_ = get_beams(ai,ai)
        kcoadd = np.nan_to_num(kcoadd/maps.gauss_beam(modlmap,ifwhm))
        ncoadd = np.nan_to_num(kcoadd/np.sqrt(np.mean(window**2.))) # NEED TO RETHINK THE SPATIAL WEIGHTING
        kcoadds.append(ncoadd) 
    kcoadds = enmap.enmap(np.stack(kcoadds),wcs)
    wins = enmap.enmap(np.stack(wins),wcs)
    return kcoadds,wins




def get_splits(ai):
    """
    Should also beam deconvolve and low pass here
    ai is index of array (in the "arrays" list that you specified as an argument)
    Return (nsplits,Ny,Nx) ndmap
    """
    nsplits = get_nsplits(ai)
    ksplits = []
    wins = []
    mask = enmap.read_map(xmaskfname()) # steve's mask
    for i in range(nsplits):
        # iwin = enmap.read_map(sinvvarfname(ai,i))
        iwin = 1. # window function is 1
        window = mask*iwin
        wins.append(window)
        imap = enmap.read_map(splitfname(ai,i),sel=np.s_[0,:,:]) # document sel usage
        _,_,ksplit = fc.power2d(window*imap)
        ksplits.append(ksplit)
    ksplits = enmap.enmap(np.stack(ksplits),wcs)
    wins = enmap.enmap(np.stack(wins),wcs)
    return ksplits,wins

