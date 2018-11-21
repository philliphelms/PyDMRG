import numpy as np
import mps_opt

x = mps_opt.MPS_OPT(N=10,
                    hamType='tasep',
                    maxBondDim=20,
                    tol = 1e-8,
                    maxIter = 5,
                    mpsFilename='myMPS',
                    hamParams=(0.35,-1.,2./3.))
E = x.kernel()

x = mps_opt.MPS_OPT(N=10,
                    hamType='tasep',
                    maxBondDim=20,
                    tol=1e-8,
                    maxIter=5,
                    mpsFilename='myMPS2',
                    initialGuess='data/dmrg/myMPS.npz',
                    hamParams=(0.35,-1.+0.0001,2./3.))
E2 = x.kernel()

