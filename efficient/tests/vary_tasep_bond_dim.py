import numpy as np
import time
try:
    from pydmrg.efficient import mps_opt
except:
    from PyDMRG.efficient import mps_opt

#-----------------------------------------------------------------------------
# For a single point in the TASEP phase space, we vary the max bond dimension
# to analyze how the error converges as a function of bond dimension size.
#-----------------------------------------------------------------------------

def run_test():
    N = 20
    bondDimVec = np.array([10,20,30,40])
    Evec = np.zeros(len(bondDimVec))
    diffVec = np.zeros(len(bondDimVec))
    for i in range(len(bondDimVec)):
        x = mps_opt.MPS_OPT(N=int(N),
                            maxBondDim = bondDimVec[i],
                            tol = 1e-3,
                            verbose = 0,
                            hamParams = (0.35,-1,2/3))
        Evec[i] = x.kernel()
    diffVec = np.abs(Evec-Evec[-1])
    return diffVec
