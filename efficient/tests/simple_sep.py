import numpy as np
import time
from pydmrg.efficient import mps_opt

#-----------------------------------------------------------------------------
# A simple calculation using the general sep instead of the tasep. This
# is initially set up to run the case identical to the one done in the 
# 01_simple_tasep.py example.
#-----------------------------------------------------------------------------

def run_test():
    N = 10
    x1 = mps_opt.MPS_OPT(N=N,
                        hamType = "sep",
                        hamParams = (2/3,0,1,0,0,0.35,-1))
    x2 = mps_opt.MPS_OPT(N=N,
                         hamType = "sep",
                         hamParams = (0,0.35,0,1,2/3,0,1))
    return x1.kernel(),x2.kernel()
