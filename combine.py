import tilec
import argparse, yaml
import numpy as np
from sotools import enmap
from orphics import maps,io

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

def coaddfname(aindex): return darrays[arrays[aindex]]['froot'] + darrays[arrays[aindex]]['coadd']
def splitfname(aindex,split): return darrays[arrays[aindex]]['froot'] + darrays[arrays[aindex]]['splits'] % split
def sinvvarfname(aindex,split): return darrays[arrays[aindex]]['froot'] + darrays[arrays[aindex]]['sinvvars'] % split
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
        _,_,ksplit = fc.power2d(window*enmap.read_map(splitfname(ai,i),sel=np.s_[0,:,:]))
        ksplits.append(ksplit)
    ksplits = enmap.enmap(np.stack(ksplits),wcs)
    wins = enmap.enmap(np.stack(wins),wcs)
    return ksplits,wins
    

def ncalc(ai,aj):
    """
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
    return enmap.enmap(scov,wcs),enmap.enmap(ncov,wcs)

    


Scov = np.zeros((narrays,narrays,Ny,Nx))
Ncov = np.zeros((narrays,narrays,Ny,Nx))

for aindex1 in range(narrays):
    for aindex2 in range(aindex1,narrays) :

        print("Noise calc...")
        scov,ncov = ncalc(aindex1,aindex2)
        print("Signal avg...")
        
        Scov[aindex1,aindex2] = tilec.signal_average(scov,bin_width=200)
        if aindex1!=aindex2: Scov[aindex2,aindex1] = Scov[aindex1,aindex2].copy()
        print("Noise avg...")

        Ncov[aindex1,aindex2],_ = tilec.noise_average(ncov)
        if aindex1!=aindex2: Ncov[aindex2,aindex1] = Ncov[aindex1,aindex2].copy()

        # io.plot_img(np.log10(np.fft.fftshift(scov)),"scov.png",aspect='auto')
        io.plot_img(np.log10(np.fft.fftshift(Scov[aindex1,aindex2])),"dscov.png",aspect='auto')
        # io.plot_img(np.log10(np.fft.fftshift(ncov)),"ncov.png",aspect='auto',lim=[-5,1])
        io.plot_img(np.log10(np.fft.fftshift(Ncov[aindex1,aindex2])),"dncov.png",aspect='auto',lim=[-5,1])
        io.plot_img(np.log10(np.fft.fftshift(Scov[aindex1,aindex2]+Ncov[aindex1,aindex2])),"dsncov.png",aspect='auto',lim=[-5,1])
        # io.plot_img(np.log10(np.fft.fftshift(ncov/Ncov[aindex1,aindex2])),"rcov.png",aspect='auto',lim=[-5,1])
        


Cov = Scov + Ncov
np.save("cov.npy",Cov)

# Cinv = np.linalg.inv(Cov)
# kmaps = fft.fft(coadds)
# okmap = maps.gilc(kmaps,Cinv,response_a,response_b)
