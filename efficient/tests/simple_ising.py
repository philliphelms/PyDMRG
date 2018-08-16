import numpy as np
import time
try:
    from pydmrg.efficient import mps_opt
except:
    from PyDMRG.efficient import mps_opt

#-----------------------------------------------------------------------------
# Using the Ising Hamiltonian, perform a simple steady state calculation
#-----------------------------------------------------------------------------

def run_test():
    N = 10
    x = mps_opt.MPS_OPT(N=N,
                        hamType = "ising",
                        periodic_x = True,
                        hamParams = (1,0))
    return x.kernel()