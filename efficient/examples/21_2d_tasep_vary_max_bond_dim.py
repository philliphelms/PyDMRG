import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# For the 2D sep, we set up TASEPs accross the 2D space, then compare the results
# of these to a calculations on the 1D TASEP. We vary the maximum bond 
# dimension for all of the calculations to determine what the needed bond 
# dimension might be.
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

N = 12
bondDimVec = [2,4,6,8,10,12,14,16,18,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
# Run 1D Calculation for comparison
Evec_1d = np.zeros(len(bondDimVec))
diffVec = np.zeros(len(bondDimVec))
print('Running 1D Calculations')
x = mps_opt.MPS_OPT(N=N,
                    maxBondDim = bondDimVec,
                    hamParams = (0.35,-1,2/3),
                    verbose=2,
                    hamType = 'tasep')
x.kernel()
Evec_1d = x.bondDimEnergies
print(Evec_1d)
# Run exact Diagonalization for 1D
print('Running Exact Diagonalization (1D)')
E_ed = x.exact_diag()
# Run mean field 1d
print('Running mean field (1D)')
E_mf = x.mean_field()
# Run 2D in opposite direction
Evec_2d_notaligned = np.zeros(len(bondDimVec))
print('Running misaligned 2D calculations')
x = mps_opt.MPS_OPT(N=N**2,
                    maxBondDim = bondDimVec,
                    hamType="sep_2d",
                    plotExpVals=False,
                    plotConv=False,
                    verbose = 2,
                    hamParams = (0,0,0,0,0,0,
                                 1,0,0,0.35,2/3,0,-1))
x.kernel()
Evec_2d_notaligned = x.bondDimEnergies/N
# Run 2D in aligned direction
Evec_2d_aligned = np.zeros(len(bondDimVec))
print('Running aligned 2D calculations')
x = mps_opt.MPS_OPT(N=N**2,
                    maxBondDim = bondDimVec,
                    hamType="sep_2d",
                    plotExpVals=False,
                    plotConv=False,
                    verbose=2,
                    hamParams = (0,1,0.35,0,0,2/3,      # jl,jr,il,ir,ol,or,
                                 0,0,0,   0,0,0  ,-1))  # ju,jd,it,ib,ot,ob,s
x.kernel()
Evec_2d_aligned = x.bondDimEnergies/N
# Calculate Errors
err_mf = np.abs(E_mf-E_ed)
print(err_mf)
errVec_1d = np.abs(Evec_1d-E_ed)
errVec_2d_aligned = np.abs(Evec_2d_aligned-E_ed)
errVec_2d_notaligned = np.abs(Evec_2d_notaligned-E_ed)
# Create Plot
fig1 = plt.figure()
plt.semilogy(np.array([np.min(bondDimVec),np.max(bondDimVec)]),np.array([err_mf,err_mf]),':',linewidth=3)
plt.semilogy(bondDimVec,errVec_1d,linewidth=3)
plt.semilogy(bondDimVec,errVec_2d_aligned,linewidth=3)
plt.semilogy(bondDimVec,errVec_2d_notaligned,linewidth=3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Bond Dimension',fontsize=20)
plt.ylabel('$E-E_{exact}$',fontsize=20)
plt.legend(('Mean Field','1D DMRG','2D DMRG (aligned)','2D DMRG (not aligned)'))
fig1.savefig('varyMaxBondDim_'+str(bondDimVec[-1])+'.pdf')
