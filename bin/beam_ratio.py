from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from soapack import interfaces as sints

ells = np.arange(2,15000,1)
dm = sints.ACTmr3()

def add(bs,col):
    for i,b in enumerate(bs):
        if i==0:
            denom = dm.get_beam( ells, *b,  kind='normalized',sanitize=False,sanitize_val=1e-3)
            continue
        r = dm.get_beam( ells, *b,  kind='normalized',sanitize=False,sanitize_val=1e-3) / denom
        _,_,array = b
        pl.add(ells,r,color=col,label=array if i==1 else None)



pl = io.Plotter(xyscale='linlin',xlabel='$\\ell$',ylabel='$B/B_0$')
bs = [("s14","deep56","pa1_f150"),("s13","deep5","pa1_f150"),("s13","deep6","pa1_f150"),("s15","deep56","pa1_f150"),("s13","deep1","pa1_f150"),("s15","boss","pa1_f150")]
add(bs,'C0')
bs = [("s14","deep56","pa2_f150"),("s15","deep56","pa2_f150"),("s15","boss","pa2_f150")]
add(bs,'C1')
bs = [("s15","deep56","pa3_f150"),("s15","boss","pa3_f150")]
add(bs,'C2')
bs = [("s15","deep56","pa3_f090"),("s15","boss","pa3_f090")]
add(bs,'C3')
pl.done('beamrat.png')

