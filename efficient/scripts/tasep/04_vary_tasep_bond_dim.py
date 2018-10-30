import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# For a single point in the TASEP phase space, we vary the max bond dimension
# to analyze how the error converges as a function of bond dimension size.
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

N = 10
bondDimVec = np.array([1,2,3,4,5,6,7,8,9,10])#,11,12,13,14,15,16,17,18,19,20,30,40,50])
Evec = np.zeros(len(bondDimVec))
diffVec = np.zeros(len(bondDimVec))
for i in range(len(bondDimVec)):
    print('\tRunning Calcs for M = {}'.format(bondDimVec[i]))
    x = mps_opt.MPS_OPT(N=int(N),
                        maxBondDim = bondDimVec[i],
                        tol = 1e-1,
                        hamParams = (0.35,-1,2/3))
    Evec[i] = x.kernel()
diffVec = np.abs(Evec-Evec[-1])
fig = plt.figure()
plt.semilogy(bondDimVec,diffVec,'b-',linewidth=5)
plt.semilogy(bondDimVec,diffVec,'ro',markersize=10)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Bond Dimension',fontsize=20)
plt.ylabel('$E-E_{exact}$',fontsize=20)
fig.savefig('varyMaxBondDim.pdf')
