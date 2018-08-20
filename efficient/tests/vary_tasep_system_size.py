import numpy as np
import time
from pydmrg.efficient import mps_opt

#-----------------------------------------------------------------------------
# Calculations at a single point in phase space for the tasep, where we 
# increase the system size slowly and work towards the thermodynamic
# limit.
#-----------------------------------------------------------------------------

def run_test():
    N_vec = np.array([2,4,6,8,10,14,18,22])
    s = np.array([-1,1])
    current = np.zeros(len(N_vec))
    for i in range(len(N_vec)):
        print('N = {}'.format(N_vec[i]))
        N = int(N_vec[i])
        x1 = mps_opt.MPS_OPT(N=N,
                            hamType='tasep',
                            maxBondDim = 50,
                            tol = 1e-3,
                            hamParams=(3/5,s[0],2/3))
        print('\ts = {}'.format(s[0]))
        E_left = x1.kernel()
        x2 = mps_opt.MPS_OPT(N=N,
                            hamType='tasep',
                            maxBondDim = 50,
                            tol = 1e-3,
                            hamParams=(3/5,s[1],2/3))
        print('\ts = {}'.format(s[1]))
        E_right = x2.kernel()
        current[i] = (E_right-E_left)/(s[1]-s[0])/(N+1)
    return current

if __name__ == "__main__":
    run_test()
