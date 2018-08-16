import numpy as np
import time
try:
    from pydmrg.efficient import mps_opt
except:
    from PyDMRG.efficient import mps_opt

#-----------------------------------------------------------------------------
# Calculations at a single point in phase space for the tasep, where we 
# increase the system size slowly and work towards the thermodynamic
# limit.
#-----------------------------------------------------------------------------

def run_test():
    N_vec = np.array([10,15,20])
    s = np.array([-0.01,0.01])
    current = np.zeros(len(N_vec))
    for i in range(len(N_vec)):
        N = int(N_vec[i])
        #print('Running Calcs for N={}'.format(N))
        x1 = mps_opt.MPS_OPT(N=N,
                            hamType='tasep',
                            periodic_x=False,
                            add_noise = False,
                            maxBondDim = 50,
                            tol = 1e-3,
                            hamParams=(3/5,s[0],2/3))
        E_left = x1.kernel()
        x2 = mps_opt.MPS_OPT(N=N,
                            hamType='tasep',
                            periodic_x=False,
                            add_noise=False,
                            maxBondDim = 50,
                            tol = 1e-3,
                            hamParams=(3/5,s[1],2/3))
        E_right = x2.kernel()
        current[i] = (E_right-E_left)/(s[1]-s[0])/(N+1)
    return current
