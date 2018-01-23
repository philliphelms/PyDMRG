import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# This script changes the maximum bond dimension to determine how convergence
# depends on this. It is for the TASEP system
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

N = np.array([4,6,8,10])
bondDimVec = np.array([1,2,3,4,10])
col_vec = ['r','y','g','b','c','k','m']
fig1 = plt.figure()
for j in range(len(N)):
    Evec = np.zeros(len(bondDimVec))
    diffVec = np.zeros(len(bondDimVec))
    for i in range(len(bondDimVec)):
        print('\tRunning Calcs for M = {}'.format(bondDimVec[i]))
        x = mps_opt.MPS_OPT(N=int(N[j]),
                            maxBondDim = bondDimVec[i],
                            tol = 1e-1,
                            hamParams = (0.35,-1,2/3))
        Evec[i] = x.kernel()
    #E_ed = x.exact_diag()
    E_mf = x.mean_field()
    #diffVec = np.abs(Evec-E_exact)
    diffVec = np.abs(Evec-Evec[-1])
    plt.semilogy(bondDimVec[:-1],diffVec[:-1],col_vec[j]+'-o',linewidth=5,markersize=10,markeredgecolor='k')
    plt.semilogy([bondDimVec[0],bondDimVec[-2]],[np.abs(E_mf-Evec[-1]),np.abs(E_mf-Evec[-1])],col_vec[j]+':',linewidth=5)
    #plt.plot(np.array([bondDimVec[0],bondDimVec[-1]]),np.array([0,0]),'b--',linewidth=5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
plt.xlabel('Bond Dimension',fontsize=20)
plt.ylabel('$E-E_{exact}$',fontsize=20)
plt.legend(('DMRG','Mean Field'))
fig1.savefig('varyMaxBondDim.pdf')
