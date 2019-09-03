from tilec import fg as tfg
import numpy as np
import glob
import pickle
from tilec.utils import get_test_fdict

version = "data/MM_20190903"


def test(version):
    fdict = get_test_fdict()
    fdict0 = pickle.load(open("%s_fdict.pkl" % version,'rb'))
    for key in fdict0['mix0'].keys():
        assert np.all(np.isclose(fdict0['mix0'][key],fdict['mix0'][key]))

    for comp in fdict0['mix1'].keys():
        for key in fdict0['mix1'][comp].keys():
            assert np.isclose(fdict0['mix1'][comp][key],fdict['mix1'][comp][key])

        
test(version)
