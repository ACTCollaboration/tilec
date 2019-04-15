from orphics import maps,io,cosmology,stats
from pixell import enmap,wcsutils
import numpy as np
import os,sys



# Get the global config
gconfig = io.config_from_yaml("input/data.yml")

class Empty:
    def process(self,ncomp=1,srcfree=True,skip_splits=False,pnormalize=True):
        """
        ai is index of array (in the "arrays" list that you specified as an argument)
        Return (nsplits,Ny,Nx) fourier transform
        Return (Ny,Nx) fourier transform of coadd
        """
        if ncomp!=1: raise NotImplementedError
        
        nsplits = self.nsplits
        ksplits = []
        imaps = []
        wins = []
        for i in range(nsplits):
            iwin = self.get_split_ivar(i,ncomp,srcfree)
            wins.append(iwin.copy())
            imap = self.get_split(i,ncomp,srcfree) * iwin
            imaps.append(imap.copy())
            if not(skip_splits):
                _,_,ksplit = self.fc.power2d(imap*self.mask)
                ksplits.append(ksplit.copy()/ np.sqrt(np.mean((iwin*self.mask)**2.)))
        if not(skip_splits): ksplits = enmap.enmap(np.stack(ksplits),self.wcs)
        wins = enmap.enmap(np.stack(wins),self.wcs)
        imaps = enmap.enmap(np.stack(imaps),self.wcs)
        coadd = np.nan_to_num(imaps.sum(axis=0)/wins.sum(axis=0))
        kcoadd = enmap.enmap(self.fc.fft(coadd*self.mask),self.wcs)
        if pnormalize: kcoadd = kcoadd / np.sqrt(np.mean(self.mask**2.))
        return ksplits,kcoadd
    
    
