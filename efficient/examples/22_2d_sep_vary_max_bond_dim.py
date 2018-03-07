import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt
from sys import argv

#-----------------------------------------------------------------------------
# This is the same type of calculation as in example 21, but we are now working
# with a symmetric SEP instead of the totally assymetric SEP.
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'


#N = int(argv[1])
#bondDimVec = int(argv[2])
N = 6
bondDimVec = 100#[100,200,300,400,500,600,700,800,900,1000]
tol = 1e-10
maxIter = 5
maxEigIter = 5
#tol = [1e-1]*(len(bondDimVec)-1)
#tol.insert(-1,1e-10)
#maxIter = [2]*(len(bondDimVec)-1)
#maxIter.insert(-1,10)

# Run 1D Calculation for comparison

x = mps_opt.MPS_OPT(N=N,
                    maxBondDim = bondDimVec,
                    hamType    = "sep",
                    hamParams  = (0.9,0.1,0.5,0.5,0.1,0.9,-1))
x.kernel()
Evec_1d = x.bondDimEnergies
E_ed = x.exact_diag()
E_mf = x.mean_field()

# Run 2D in opposite direction
print('\nRun 2D - Not Aligned\n')
x = mps_opt.MPS_OPT(N          = [N,N],
                    maxBondDim = bondDimVec,
                    hamType    ="sep_2d",
                    maxIter    = maxIter,
                    max_eig_iter = maxEigIter,
                    add_noise = True,
                    verbose = 4,
                    tol        = tol,
                    hamParams  = (0,0,0,0,0,0,
                                  0.5,0.5,0.9,0.1,0.1,0.9,-1))
x.kernel()
Evec_2d_notaligned = x.bondDimEnergies/N
