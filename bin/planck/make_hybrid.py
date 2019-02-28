import argparse
parser = argparse.ArgumentParser()
parser.add_argument("Tfile")
parser.add_argument("Pfile")
parser.add_argument("ofile")
args = parser.parse_args()
import numpy as np
from enlib import enmap

T = enmap.read_map(args.Tfile, sel=(0,))
omap = enmap.zeros((3,)+T.shape, T.wcs, T.dtype)
print(omap.shape)
omap[0] = T
omap[1:] = enmap.read_map(args.Pfile, sel=(slice(1,3),))
enmap.write_map(args.ofile, omap)
