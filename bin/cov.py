from tilec import datamodel,ilc

"""

This script produces an empirical covariance matrix
from Planck and ACT data.


"""

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("version", type=str,help='Region name.')
parser.add_argument("region", type=str,help='Region name.')
parser.add_argument("arrays", type=str,help='Comma separated list of array names. Array names map to a data specification in data.yml')
# parser.add_argument("--nsim",     type=int,  default=None,help="A description.")
# parser.add_argument("-N", "--nsim",     type=int,  default=None,help="A description.")
# parser.add_argument("-f", "--flag", action='store_true',help='A flag.')
args = parser.parse_args()

gconfig = datamodel.gconfig

kcoadds = []
ksplits = []
lmins = []
lmaxs = []
anisotropic_pairs = []
for array in args.arrays.split(','):
    array_datamodel = gconfig[array]['data_model']
    dm = datamodel.datamodels[array_datamodel](args.region,gconfig[array])
    ksplit,kcoadd = dm.process()
    kcoadds.append(kcoadd.copy())
    ksplits.append(ksplit.copy())
    lmins.append(dm.c['lmin'])
    lmaxs.append(dm.c['lmax'])
    anisotropic_pairs.append(dm.c['hybrid_average'])



signal_bin_width=80
signal_interp_order=0
dfact=16
rfit_bin_width = 80
rfit_wnoise_width=250
rfit_lmin=300

Cov = ilc.build_empirical_cov(ksplits,kcoadds,lmins,lmaxs,
                        anisotropic_pairs,
                        signal_bin_width=signal_bin_width,
                        signal_interp_order=signal_interp_order,
                        dfact=(dfact,dfact),
                        rfit_lmaxes=None,
                        rfit_wnoise_width=rfit_wnoise_width,
                        rfit_lmin=rfit_lmin,
                        rfit_bin_width=None,
                        fc=dm.fc,return_full=False,
                        verbose=True,
                        debug_plots_loc=save_loc)

enmap.write_map("%s/datacov_triangle.hdf" % save_loc,Cov.data)
