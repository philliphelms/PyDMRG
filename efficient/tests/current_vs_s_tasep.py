import numpy as np
import time
from pydmrg.efficient import mps_opt

#-----------------------------------------------------------------------------
# For the TASEP model, this script calculations the current and cumulant 
# generating function as a function of s. This is for a single value of alpha
# and beta, which are specified on line 39.
#-----------------------------------------------------------------------------

def run_test():
    # Run TASEP Current Calculations
    N_vec = np.array([10])
    s_vec = np.linspace(-1,1,5)
    col_vec = ['r','r','y','g','b','c','k','m']
    for j in range(len(N_vec)):
        N = N_vec[j]
        Evec = np.zeros(s_vec.shape)
        Evec_adj = np.zeros(s_vec.shape)
        EE = np.zeros(s_vec.shape)
        for i in range(len(s_vec)):
            x = mps_opt.MPS_OPT(N =int(N),
                                verbose = 0,
                                hamType = "tasep",
                                hamParams = (0.35,s_vec[i],2/3))
            Evec[i] = x.kernel()
            Evec_adj[i] = Evec[i]/(N+1)
            EE[i] = x.entanglement_entropy[int(N/2)]
        Ediff = Evec[1:]-Evec[:len(Evec)-1]
        Sdiff = s_vec[1:]-s_vec[:len(s_vec)-1]
        slope = -Ediff/(Sdiff)
    return slope
