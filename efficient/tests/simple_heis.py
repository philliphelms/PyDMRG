import numpy as np
import time
from pydmrg.efficient import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# A simple calculation for the 1D heisenberg model.
#-----------------------------------------------------------------------------

def run_test():
    N = 10
    x = mps_opt.MPS_OPT(N=int(N),
                        hamType = "heis",
                        periodic_x = True,
                        hamParams = (1,0))
    return x.kernel()
