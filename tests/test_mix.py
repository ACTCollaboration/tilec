from tilec import fg as tfg
import numpy as np
import glob
import pickle
import os
dirname = os.path.dirname(os.path.abspath(__file__))

version = dirname + "/data/MM_20200307"


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def test_fg_mix():
    fdict = tfg.get_test_fdict()
    fdict0 = load_pickle("%s_fdict.pkl" % version)
    for key in fdict0['mix0'].keys():
        #print("mix0",key)
        assert np.all(np.isclose(fdict0['mix0'][key],fdict['mix0'][key]))

    for comp in fdict0['mix1'].keys():
        for key in fdict0['mix1'][comp].keys():
            #print("mix1",comp,key)
            #print(fdict0['mix1'][comp][key],fdict['mix1'][comp][key])
            assert np.isclose(fdict0['mix1'][comp][key],fdict['mix1'][comp][key])


def test_conversions():
    assert np.isclose(tfg.ItoDeltaT(545)/1e26,0.017483363768883677)
    for f in np.geomspace(1,1000,1000):
        assert np.isclose(tfg.ItoDeltaT(f),1/tfg.dBnudT(f))
        
