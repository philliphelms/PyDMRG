import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt
#from sys import argv

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
N = 12
bondDimVec = 300
tol = 1e-10
maxIter = 10
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
#E_ed = x.exact_diag()
#E_mf = x.mean_field()

# Run 2D in opposite direction
print('\nRun 2D - Not Aligned\n')
x = mps_opt.MPS_OPT(N          = N**2,
                    maxBondDim = bondDimVec,
                    hamType    ="sep_2d",
                    maxIter    = maxIter,
                    max_eig_iter = 20,
                    verbose = 6,
                    tol        = tol,
                    hamParams  = (0,0,0,0,0,0,
                                  0.5,0.5,0.9,0.1,0.1,0.9,-1))
x.kernel()
Evec_2d_notaligned = x.bondDimEnergies/N
"""
# Run 2D in aligned direction
print('\nRun 2D - Aligned\n')
x = mps_opt.MPS_OPT(N=N**2,
                    maxBondDim = bondDimVec,
                    hamType    = "sep_2d",
                    verbose    = 3,
                    maxIter    = maxIter,
                    tol        = tol,
                    hamParams  = (0.5,0.5,0.9,0.1,0.1,0.9,      # jl,jr,il,ir,ol,or,
                                 0,0,0,   0,0,0  ,-1))         # ju,jd,it,ib,ot,ob,s
x.kernel()
Evec_2d_aligned = x.bondDimEnergies/N

# Calculate Errors
err_mf = np.abs(E_mf-E_ed)
errVec_1d = np.abs(Evec_1d-E_ed)
errVec_2d_aligned = np.abs(Evec_2d_aligned-E_ed)
errVec_2d_notaligned = np.abs(Evec_2d_notaligned-E_ed)
# Create Plot
fig1 = plt.figure()
plt.semilogy(np.array([np.min(bondDimVec),np.max(bondDimVec)]),np.array([err_mf,err_mf]),':',linewidth=3)
plt.semilogy(bondDimVec,errVec_1d,linewidth=3)
#plt.semilogy(bondDimVec,errVec_2d_aligned,linewidth=3)
plt.semilogy(bondDimVec,errVec_2d_notaligned,linewidth=3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Bond Dimension',fontsize=20)
plt.ylabel('$E-E_{exact}$',fontsize=20)
plt.legend(('Mean Field','1D DMRG','2D DMRG (aligned)','2D DMRG (not aligned)'))
fig1.savefig('varyMaxBondDim_'+str(bondDimVec[-1])+'.pdf')
"""
