import numpy as np
import time
try:
    from pydmrg.efficient import mps_opt
except:
    from PyDMRG.efficient import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Run a simple calculation for a 2D Heisenberg Model
# DOESN'T WORK CORRECTLY!!!
#-----------------------------------------------------------------------------

def run_test():
    x = mps_opt.MPS_OPT(N=[3,3],
                        hamType = "heis_2d",
                        periodic_x = True,
                        periodic_y = False,
                        maxBondDim=50,
                        hamParams = (1,0))
    return x.kernel()