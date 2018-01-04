import numpy as np
import mps_opt

N = 2
s = -2
x = mps_opt.MPS_OPT(N=N,
                    hamType="sep",
                    hamParams=(0.9,0.1,0.5,0.5,0.9,0.1,-1))
x.kernel()
print(x.mpo.return_full_ham())
x.exact_diag()

