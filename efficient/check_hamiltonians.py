import numpy as np
import mps_opt

N = 4
s = -2
x = mps_opt.MPS_OPT(N=N,
                    hamType="tasep",
                    hamParams=(0.35,s,2/3))
x.kernel()
print(x.mpo.return_full_ham())
x.exact_diag()

