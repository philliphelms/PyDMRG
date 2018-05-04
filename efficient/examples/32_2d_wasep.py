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
s = np.linspace(-19.3,-18.5,100)
CGF_dmrg = np.zeros(s.shape)
for i in range(len(s)):
    x = mps_opt.MPS_OPT(N = [N,N],
                        hamType = "sep_2d",
                        #periodic_x = True,
                        periodic_y = True,
                        maxBondDim = [10,100],
                        verbose = 3,
                        maxIter = 2,
                        #plotExpVals = True,
                        #plotConv = True,
                        #hamParams = (qx,px,1/2,1/2,0,0,0,0,0,0,0,0,[s[i]/N,0]))
                        hamParams = (1/2,1/2,qx,px,1/2,1/2,0,0,1/2,1/2,0,0,[0,s[i]/N]))
                        #hamParams = (1/2,1/2,qx,px,0,0,0,0,0,0,0,0,[0,s[i]/N]))
    print('Performing Calculation for s = {}'.format(s[i]))
    CGF_dmrg[i] = x.kernel()
    print('Final Density Profile = \n{}'.format(x.calc_occ))
