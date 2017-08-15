import numpy as np
import time
import mps_dmrg

# Settings
N = 14
d = 2
D = 8
tol = 1e-3
max_sweep_cnt = 20
ham_type = "heis"
ham_params = (-1,0)
# Run Ground State Calculations
t0 = time.time()
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)
x = mps_dmrg.MPS_DMRG(L = N,
                 d = d,
                 D = D,
                 tol = tol,
                 max_sweep_cnt = max_sweep_cnt,
                 ham_type = ham_type,
                 ham_params = ham_params)
x.calc_ground_state()
t1 = time.time()
print(('#'*75+'\nTotal Time: {}\n'+'#'*75).format(t1-t0))
