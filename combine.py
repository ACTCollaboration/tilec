import tilec
import argparse, yaml
import numpy as np
from pixell import enmap
from orphics import maps,io,stats

"""
Issues:
1. SOLVED by only using ells<lmax: 12 arrays form a 22 GB covmat for deep56. Can reduce resolution by 4x if lmax=5000, fit it in 5.5 GB.
2. Cross-covariances are noisy.
3. Beam is gaussian now.
4. Maps will be band-limited. Maps will require inpainting.

"""


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
def splitfname(aindex,split):
    try: splitstart = darrays[arrays[aindex]]['splitstart']
    except: splitstart = 0
    return darrays[arrays[aindex]]['froot'] + darrays[arrays[aindex]]['splits'] % (split + splitstart)
def sinvvarfname(aindex,split):
    try: splitstart = darrays[arrays[aindex]]['splitstart']
    except: splitstart = 0
    return darrays[arrays[aindex]]['froot'] + darrays[arrays[aindex]]['sinvvars'] % (split + splitstart)
def get_nsplits(aindex): return darrays[arrays[aindex]]['nsplits']
def xmaskfname(aindex): return darrays[arrays[aindex]]['xmask']

narrays = len(arrays)
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
    mask = enmap.read_map(xmaskfname(ai))
    for i in range(nsplits):
        window = mask*enmap.read_map(sinvvarfname(ai,i))
        wins.append(window)
        imap = enmap.read_map(splitfname(ai,i),sel=np.s_[0,:,:])
        _,_,ksplit = fc.power2d(window*imap)
        ksplits.append(ksplit)
    ksplits = enmap.enmap(np.stack(ksplits),wcs)
    wins = enmap.enmap(np.stack(wins),wcs)
    return ksplits,wins
    

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
        dncov,_ = tilec.noise_average(ncov,radial_fit=do_radial) if ((aindex1==aindex2) or is90150(aindex1,aindex2)) else (0.,None)

        if isplanck(aindex1) and isplanck(aindex2):
            lmin = maps.minimum_ell(shape,wcs)
        else:
            lmin = 300
            
        sel = np.logical_or(modlmap<lmin,modlmap>lmax)
        ifwhm,jfwhm = get_beams(aindex1,aindex2)
        dncov /= (maps.gauss_beam(modlmap,ifwhm)*maps.gauss_beam(modlmap,jfwhm))
        dscov /= (maps.gauss_beam(modlmap,ifwhm)*maps.gauss_beam(modlmap,jfwhm))
        dncov[sel] = np.inf
        dscov[sel] = np.inf

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

# Cinv = np.linalg.inv(Cov)
# kmaps = fft.fft(coadds)
# okmap = maps.gilc(kmaps,Cinv,response_a,response_b)
