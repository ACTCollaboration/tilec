from tilec import fg as tfg
import numpy as np
import glob
import pickle

version = "data/MM_20190903"


def test(version):
    fdict = tfg.get_test_fdict()
    fdict0 = pickle.load(open("%s_fdict.pkl" % version,'rb'))
    for key in fdict0['mix0'].keys():
        #print("mix0",key)
        assert np.all(np.isclose(fdict0['mix0'][key],fdict['mix0'][key]))

    for comp in fdict0['mix1'].keys():
        for key in fdict0['mix1'][comp].keys():
            #print("mix1",comp,key)
            #print(fdict0['mix1'][comp][key],fdict['mix1'][comp][key])
            assert np.isclose(fdict0['mix1'][comp][key],fdict['mix1'][comp][key])

        
test(version)
