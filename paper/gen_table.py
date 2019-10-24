from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys

fname = 'paper/compsep-table - Sheet1.csv'

data = np.genfromtxt(fname, delimiter=',', dtype=str)
Ncols = data.shape[1]
Nrows = data.shape[0]
Nells = '|'.join(['l']*Ncols)

output = \
    """\\begin{table}[ht]
\\label{tab:maps}
\\centering
\\caption{Description of maps used for component separation. The central frequencies are not intended to be precise; we use the full bandpass in our analysis. While the beam FWHM for Planck reflects what we assume in our analysis, for ACT, only rough estimates are shown in this table; we use the appropriate harmonic transfer function in our analysis.}
\\begin{tabular}[t]{%s} 
""" % Nells


for i,line in enumerate(data):
    row = ' & '.join(['\\textbf{%s}' % l for l in line] if i==0 else line).replace('_','\_')
    bstr = " \\\\" if i!=(Nrows-1) else ""
    if i==1: bstr = bstr + " \\hline"
    output = output + row + bstr + " \n"

    
output = output + \
    """\end{tabular}
\end{table}
"""

print(output)
