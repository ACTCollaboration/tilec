from tilec import fg as tfg
import numpy as np
import glob
import pickle
from tilec.utils import get_test_fdict

version = "MM_20190903"

fdict = get_test_fdict()
pickle.dump(fdict,open("%s_fdict.pkl" % version,'wb'))
