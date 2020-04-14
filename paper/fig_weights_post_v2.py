from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from soapack import interfaces as sints
from tilec import utils as tutils
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib import scale as mscale
from matplotlib import scale as mscale
from matplotlib.transforms import Transform


def _mask_nonpos(a):
    """
Return a Numpy masked array where all non-positive 1 are
masked. If there are no non-positive, the original array
is returned.
"""
    mask = a <= 0.0
    if mask.any():
        return ma.MaskedArray(a, mask=mask)
    return a


class OPointSixScale(mscale.ScaleBase):
    """
    Scale used by the Planck collaboration to plot Temperature power spectra:
    base-10 logarithmic up to l=50, and linear from there on.
    
    Care is taken so non-positive values are not plotted.
    """
    name = 'opointsix'

    def __init__(self, axis, **kwargs):
        pass

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(
            matplotlib.ticker.LogLocator(base = 10, subs = [2,5]))
        axis.set_minor_locator(
            matplotlib.ticker.LogLocator(base = 10, subs = [1,2,3,4,5,6,7,8,9]))
                # np.concatenate((np.arange(2, 10),
                #                 np.arange(10, 50, 10),
                    #                 np.arange(floor(change/100), 2500, 100))))


    def get_transform(self):
        """
        Return a :class:`~matplotlib.transforms.Transform` instance
        appropriate for the given logarithm base.
        """
        nonpos = "mask"
        return self.OPointSixTransform(nonpos)

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Limit the domain to positive values.
        """
        return (vmin <= 0.0 and minpos or vmin,
                vmax <= 0.0 and minpos or vmax)

    class OPointSixTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, nonpos):
            Transform.__init__(self)
            if nonpos == 'mask':
                self._handle_nonpos = _mask_nonpos
            else:
                self._handle_nonpos = _clip_nonpos

        def transform_non_affine(self, a):
            return a**(0.3)


        def inverted(self):
            return OPointSixScale.InvertedOPointSixTransform()

    class InvertedOPointSixTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def transform_non_affine(self, a):
            return a**(-0.3)

        def inverted(self):
            return OPointSixTransform()


mscale.register_scale(OPointSixScale)

cols = ["C%d" % i for i in range(30)]
cols.remove('C8')
for comp in ['cmb','comptony']:

    region = "deep56"
    qids = "d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08".split(',')

    #version = "map_v1.0.0_rc_joint"
    #cversion = "v1.0.0_rc"

    # version = "map_v1.1.1_joint"
    # cversion = "v1.1.1"

    version = "map_v1.2.0_joint"
    cversion = "v1.2.0"


    bw = 20
    bin_edges = np.arange(20,10000,bw)
    aspecs = tutils.ASpecs().get_specs


    w1ds = []

    # actmap = {"d56_01":"D1_149","d56_02":"D2_149","d56_03":"D3_149","d56_04":"D4_149","d56_05":"D5_097","d56_06":"D6_149"}
    actmap = {"d56_01":"D56_1_149","d56_02":"D56_2_149","d56_03":"D56_3_149","d56_04":"D56_4_149","d56_05":"D56_5_097","d56_06":"D56_6_149"}

    if comp=='comptony':
        wstr = '$W (1/\mu K \\times 10^7)$'
    else:
        wstr = '$W$ (dimensionless)'

    #pl = io.Plotter(xyscale='loglin',xlabel='$\\ell$',ylabel=wstr,ftsize=16)
    #pl = io.Plotter(xlabel='$\\ell$',ylabel=wstr,ftsize=16,xscale='linear',yscale='symlog',labsize=8) # !!!
    pl = io.Plotter(xlabel='$\\ell$',ylabel=wstr,ftsize=16,xscale='linear',yscale='linear',labsize=8) # !!!
    for i in range(len(qids)):
        col = cols[i]
        qid = qids[i]
        lmin,lmax,hybrid,radial,friend,cfreq,fgroup,wrfit = aspecs(qid)    


        if tutils.is_lfi(qid):
            ls = "-."
            lab = "LFI %d GHz" % cfreq 
        elif tutils.is_hfi(qid):
            ls = "--"
            lab = "HFI %d GHz" % cfreq 
        elif qid=="d56_05" or qid=="d56_06":
            ls = ":"
            aind = qid.split("_")[1]
            lab = actmap[qid] #"ACT_%s %d GHz" % (aind,cfreq )
        else:
            ls = "-"
            aind = qid.split("_")[1]
            lab = actmap[qid] #"ACT_%s %d GHz" % (aind,cfreq )
        mul = 1e7 if comp=='comptony' else 1
        cents,w1d = np.loadtxt("weights_%s_%s_%s.txt" % (comp,version,lab),unpack=True)

        pl.add(cents,w1d*mul,label=lab if comp=='comptony' else None,ls=ls,color=col)
    pl._ax.set_xlim(20+bw/2.,10000)

    #if comp=='cmb': pl._ax.set_ylim(-1,1.5) # !!!!


    pl._ax.yaxis.set_minor_locator(AutoMinorLocator())
    #pl._ax.xaxis.set_minor_locator(AutoMinorLocator())
    pl._ax.tick_params(axis='x',which='both', width=1)
    pl._ax.tick_params(axis='y',which='both', width=1)
    pl._ax.xaxis.grid(True, which='both',alpha=0.3)
    pl._ax.yaxis.grid(True, which='both',alpha=0.3)

    
    font = {'family': 'serif',
            'color':  'black',
            'weight': 'bold',
            'size': 14,
        }


    if comp=='comptony': 
        pl.legend(loc='upper right', bbox_to_anchor=(1.45, 1),labsize=12)
        pl._ax.text(600, 3, "Compton-$y$ weights",fontdict = font)
    elif comp=='cmb': 
        #pl._ax.text(600, 4, "CMB+kSZ weights",fontdict = font)
        pl._ax.text(600, 0.8, "CMB+kSZ weights",fontdict = font) # !!!

    pl._ax.set_xscale('opointsix')
    pl.done(("%s/fig_weight1d_%s_%s" % (os.environ['WORK'],comp,version)).replace('.','_')+".pdf")
