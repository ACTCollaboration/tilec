import tilec
import argparse, yaml
import numpy as np
from pixell import enmap,fft,utils
from orphics import maps,io,stats

make_cov = False

"""
Issues:
1. SOLVED by only using ells<lmax: 12 arrays form a 22 GB covmat for deep56. Can reduce resolution by 4x if lmax=5000, fit it in 5.5 GB.
2. Cross-covariances are noisy.
3. Beam is gaussian now.
4. Maps will be band-limited. Maps will require inpainting.

"""
# Set up SZ frequency dependence
def gnu(nu_ghz,tcmb=2.7255):
    nu = 1e9*np.asarray(nu_ghz)
    hplanck = 6.62607e-34
    kboltzmann = 1.38065e-23 
    x = hplanck*nu/kboltzmann/tcmb
    coth = np.cosh(x/2.)/np.sinh(x/2.)
    return x*coth-4.

# Parse command line
parser = argparse.ArgumentParser(description='Make ILC maps.')
parser.add_argument("arrays", type=str,help='List of arrays named in arrays.yml.')
args = parser.parse_args()

arrays = args.arrays.split(',')
with open("arrays.yml") as f:
    config = yaml.safe_load(f)
darrays = {}
for d in config['arrays']:
    darrays[d['name']] = d.copy()
    
narrays = len(arrays)
freqs = []
for i in range(narrays):
    f = darrays[arrays[i]]['freq']
    freqs.append(f)
    
def get_beams(ai,aj):
    return darrays[arrays[ai]]['beam'],darrays[arrays[aj]]['beam']
def isplanck(aindex):
    name = darrays[arrays[aindex]]['name'].lower()
    return True if ("hfi" in name) or ("lfi" in name) else False
def is90150(ai,aj):
    iname = darrays[arrays[ai]]['name'].lower()
    jname = darrays[arrays[aj]]['name'].lower()
    return True if (("90" in iname) and ("150" in jname)) or (("90" in jname) and ("150" in iname)) else False
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

shape,wcs = enmap.read_fits_geometry(coaddfname(0))
Ny,Nx = shape[-2:]
fc = maps.FourierCalc(shape[-2:],wcs)

def get_splits(ai):
    """
    Should also beam deconvolve and low pass here
    """
    nsplits = get_nsplits(ai)
    ksplits = []
    wins = []
    mask = enmap.read_map(xmaskfname())
    for i in range(nsplits):
        # iwin = enmap.read_map(sinvvarfname(ai,i))
        iwin = 1.
        window = mask*iwin
        wins.append(window)
        imap = enmap.read_map(splitfname(ai,i),sel=np.s_[0,:,:])
        _,_,ksplit = fc.power2d(window*imap)
        ksplits.append(ksplit)
    ksplits = enmap.enmap(np.stack(ksplits),wcs)
    wins = enmap.enmap(np.stack(wins),wcs)
    return ksplits,wins

def get_coadds():
    """
    Should also beam deconvolve and low pass here
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
        ncoadd = np.nan_to_num(kcoadd/np.sqrt(np.mean(window**2.))) # NEED TO RETHINK THE SPATIAL WEIGHTING
        kcoadds.append(ncoadd) 
    kcoadds = enmap.enmap(np.stack(kcoadds),wcs)
    wins = enmap.enmap(np.stack(wins),wcs)
    return kcoadds,wins


def ncalc(ai,aj):
    """
    Cross spectrum and noise power calculator
    For i x j element of Cov
    """
    iksplits,iwins = get_splits(ai) # each ksplit multiplied by mask and inv var map, returning also mask*inv var map
    if aj!=ai:
        jksplits,jwins = get_splits(aj) # each ksplit multiplied by mask and inv var map, returning also mask*inv var map
    else:
        jksplits = iksplits.copy()
        jwins = iwins.copy()
    nisplits = iksplits.shape[0]
    njsplits = jksplits.shape[0]
    autos = 0. ; crosses = 0.
    nautos = 0 ; ncrosses = 0
    for p in range(nisplits):
        for q in range(p,njsplits):
            if p==q:
                nautos += 1
                autos += fc.f2power(iksplits[p],jksplits[q]) / np.mean(iwins[p]*jwins[q])
            else:
                ncrosses += 1
                crosses += fc.f2power(iksplits[p],jksplits[q]) / np.mean(iwins[p]*jwins[q])
    autos /= nautos
    crosses /= ncrosses
    scov = crosses
    ncov = autos-crosses
    return enmap.enmap(scov,wcs),enmap.enmap(ncov,wcs),enmap.enmap(autos,wcs)

    

modlmap = enmap.modlmap(shape,wcs)
lmax = 5000
ells = modlmap[modlmap<lmax].reshape(-1)
nells = ells.size

if make_cov:
    Scov = np.zeros((narrays,narrays,nells))
    Ncov = np.zeros((narrays,narrays,nells))

    for aindex1 in range(narrays):
        for aindex2 in range(aindex1,narrays) :

            print("Noise calc...")
            scov,ncov,autos = ncalc(aindex1,aindex2)

            if isplanck(aindex1) and isplanck(aindex2):
                """
                Ugh Planck is so low noise at low ells that the cross spectrum
                is too close to the auto spectrum. So the inferred noise can be
                negative. So we fill in below some lmin with the mean noise between
                lmin and lmin+ldelta.
                """
                lmin = 400
                ldelta = 200
                nmean = ncov[np.logical_and(modlmap>lmin,modlmap<(lmin+ldelta))].mean()
                ncov[modlmap<lmin] = nmean
                """
                OR we can just take the absolute value below some lmin
                """
                # lmin = 600
                # ncov[modlmap<lmin] = np.abs(ncov[modlmap<lmin])

            print("Signal avg...")
            dscov = tilec.signal_average(scov,bin_width=40)

            print("Noise avg...")

            if isplanck(aindex1) or isplanck(aindex2):
                do_radial = False
            elif aindex1==aindex2:
                do_radial = True
            else:
                do_radial = False
            # dncov,_ = tilec.noise_average(ncov,radial_fit=do_radial) if ((aindex1==aindex2) or is90150(aindex1,aindex2)) else (0.,None)
            dncov,_ = tilec.noise_average(ncov,radial_fit=do_radial) if (aindex1==aindex2)  else (0.,None) # ignore 90 / 150

            if isplanck(aindex1) and isplanck(aindex2):
                lmin = maps.minimum_ell(shape,wcs)
            else:
                lmin = 300

            sel = np.logical_or(modlmap<lmin,modlmap>lmax)
            ifwhm,jfwhm = get_beams(aindex1,aindex2)
            dncov /= np.nan_to_num(maps.gauss_beam(modlmap,ifwhm)*maps.gauss_beam(modlmap,jfwhm))
            dscov /= np.nan_to_num(maps.gauss_beam(modlmap,ifwhm)*maps.gauss_beam(modlmap,jfwhm))
            dncov[sel] = 1e90 #np.inf # inf gives invertible matrix but nonsensical output, 1e90 gives noninvertible, but with eigpow sensible
            dscov[sel] = 1e90 #np.inf

            # io.plot_img((np.fft.fftshift(ncov)),"tuncov%d%d.png"%(aindex1,aindex2),aspect='auto')
            # io.plot_img(np.log10(np.fft.fftshift(scov+ncov)),"udsncov%d%d.png"%(aindex1,aindex2),aspect='auto',lim=[-5,1])
            io.plot_img(np.log10(np.fft.fftshift(ncov)),"uncov%d%d.png"%(aindex1,aindex2),aspect='auto',lim=[-5,1])
            io.plot_img(np.log10(np.fft.fftshift(scov)),"usncov%d%d.png"%(aindex1,aindex2),aspect='auto',lim=[-5,1])

            io.plot_img(np.log10(np.fft.fftshift(dscov+dncov)),"dsncov%d%d.png"%(aindex1,aindex2),aspect='auto',lim=[-5,1])
            io.plot_img(np.log10(np.fft.fftshift(dncov)),"dncov%d%d.png"%(aindex1,aindex2),aspect='auto',lim=[-5,1])
            io.plot_img(np.log10(np.fft.fftshift(dscov)),"dsncov%d%d.png"%(aindex1,aindex2),aspect='auto',lim=[-5,1])
            io.plot_img(np.log10(np.fft.fftshift(ncov/dncov)),"rcov%d%d.png"%(aindex1,aindex2),aspect='auto',lim=[-5,1])


            Scov[aindex1,aindex2] = dscov[modlmap<lmax].reshape(-1)
            Ncov[aindex1,aindex2] = dncov[modlmap<lmax].reshape(-1)


            if aindex1!=aindex2: Scov[aindex2,aindex1] = Scov[aindex1,aindex2].copy()
            if aindex1!=aindex2: Ncov[aindex2,aindex1] = Ncov[aindex1,aindex2].copy()




    Cov = Scov + Ncov
    np.save("cov.npy",Cov)

Cov = np.load("cov.npy")
Cov = np.rollaxis(Cov,2)
try:
    icinv = np.load("icinv.npy")
    #raise
    #icinv = np.linalg.inv(Cov)
except:
    print("Couldn't invert with np.linalg.inv. Trying eigpow")
    icinv = utils.eigpow(Cov,-1)
    np.save("icinv.npy",icinv)

Cinv = np.rollaxis(icinv,0,3)
ikmaps,_ = get_coadds()
kmaps = ikmaps.reshape((narrays,Ny*Nx))[:,modlmap.reshape(-1)<lmax]
bmask = maps.binary_mask(enmap.read_map(xmaskfname()))


yresponses = gnu(freqs,tcmb=2.7255)
cresponses = yresponses*0.+1.

def process(kouts,name="default",ellmax=None,y=False,y_ellmin=400):
    ellmax = lmax if ellmax is None else ellmax
    ksilc = enmap.zeros((Ny,Nx),wcs,dtype=np.complex128).reshape(-1)
    ksilc[modlmap.reshape(-1)<lmax] = np.nan_to_num(kouts.copy())
    ksilc = enmap.enmap(ksilc.reshape((Ny,Nx)),wcs)
    ksilc[modlmap>ellmax] = 0
    if y: ksilc[modlmap<y_ellmin] = 0
    msilc = np.nan_to_num(fft.ifft(ksilc,axes=[-2,-1],normalize=True).real * bmask)
    p2d = fc.f2power(ksilc,ksilc)

    bin_edges = np.arange(100,3000,40)
    binner = stats.bin2D(modlmap,bin_edges)
    cents,p1d = binner.bin(p2d)

    pl = io.Plotter(yscale='log',xlabel='l',ylabel='C')
    pl.add(cents,p1d*cents**2.)
    pl.done("p1d_%s.png" % name)
    
    
    try:
        io.plot_img(np.log10(np.fft.fftshift(p2d)),"ksilc_%s.png" % name,aspect='auto')
        io.plot_img(msilc,"msilc_%s.png" % name)
        io.plot_img(msilc,"lmsilc_%s.png" % name,lim=300)
        io.plot_img(msilc,"hmsilc_%s.png" % name,high_res=True)
    except:
        pass
    
    
iksilc = maps.silc(kmaps,Cinv)
process(iksilc,"silc_cmb")

iksilc = maps.silc(kmaps,Cinv,yresponses)
process(iksilc,"silc_y",y=True)

# iksilc = maps.cilc(kmaps,Cinv,cresponses,yresponses)
# process(iksilc,"cilc_cmb_deproj_y")

# iksilc = maps.cilc(kmaps,Cinv,yresponses,cresponses)
# process(iksilc,"cilc_y_deproj_cmb",y=True)

iksilc = maps.cilc(kmaps,Cinv,cresponses,yresponses)
process(iksilc,"cilc_cmb_deproj_y_lmax_3000",ellmax=3000)

iksilc = maps.cilc(kmaps,Cinv,yresponses,cresponses)
process(iksilc,"cilc_y_deproj_cmb_lmax_3000",ellmax=3000,y=True)

