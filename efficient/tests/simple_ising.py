import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Using the Ising Hamiltonian, perform a simple steady state calculation
#-----------------------------------------------------------------------------

def run_test():
    N = 10
    x = mps_opt.MPS_OPT(N=N,
                        hamType = "ising",
                        verbose = 0,
                        periodic_x = True,
                        hamParams = (1,0))
    return x.kernel()
