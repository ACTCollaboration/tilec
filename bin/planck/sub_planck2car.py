import os,sys,glob
ifs = glob.glob("/scratch/r/rbond/msyriac/data/planck/data/pr2/*%s*.fits" % sys.argv[1]) + glob.glob("/scratch/r/rbond/msyriac/data/planck/data/pr3/*%s*.fits" % sys.argv[1])

fs = []
for f in ifs:
    if ("_map.fits" in f) or ("_ivar.fits" in f) or ("Mask_Lensing" in f) : continue
    fs.append(f)

fstr = ' '.join(fs)
tfile = "/scratch/r/rbond/msyriac/data/planck/data/s16_template.fits"
cmd = "python planck2car.py %s %s --verbose" % (fstr,tfile)
scmd = "mpi_niagara 8 \"%s\" -t 20 --walltime 01:00:00" % cmd
os.system(scmd)
print(scmd)
