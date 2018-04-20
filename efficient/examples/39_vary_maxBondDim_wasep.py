import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt 
from sys import argv

#-----------------------------------------------------------------------------
# In this script, we use the mpo module to calculation the hamiltonian
# for various models (1D SEP, 2D SEP, 1D Heis, 2D Heis) and print each of
# these to the terminal
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)
np.set_printoptions(linewidth=100)
plt.style.use('fivethirtyeight') #'fivethirtyeight') #'ggplot'

#-----------------------------------------------------------------------------
# 2D WASEP
#-----------------------------------------------------------------------------
N=12
n_points = 10
E = 10
px = 1/2*np.exp(-E/N)
qx = 1/2*np.exp(E/N)
s = np.array([-29.1919191919,-18.6868686869,-10])
CGF_dmrg = np.zeros(s.shape)
x = np.array([])
for i in range(len(s)):
    np.insert(x,len(x),mps_opt.MPS_OPT(N = [N,N],
                                    hamType = "sep_2d",
                                    #periodic_x = True,
                                    periodic_y = True,
                                    verbose = 3,
                                    maxBondDim = 2,
                                    maxIter = 2,
                                    #plotExpVals = True,
                                    #plotConv = True,
                                    #hamParams = (qx,px,1/2,1/2,0,0,0,0,0,0,0,0,[s[i]/N,0]))
                                    hamParams = (1/2,1/2,qx,px,1/2,1/2,0,0,1/2,1/2,0,0,[0,s[i]/N])))
    CGF_dmrg[i] = x[i].kernel()
np.save('2D_wasep_data_{}x{}'.format(N,N))
plt.figure()
plt.plot(s,CGF_dmrg,'b:',label='DMRG')
plt.xlabel('$\lambda$')
plt.ylabel('$\psi$')
plt.legend()
plt.show()
plt.figure()
plt.xlabel('$\lambda$')
plt.ylabel('$\psi$')
