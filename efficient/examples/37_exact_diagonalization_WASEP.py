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

"""
#-----------------------------------------------------------------------------
# 2D WASEP
#-----------------------------------------------------------------------------
N=3
n_points = 10
E = 10
px = 1/2*np.exp(-E/N)
qx = 1/2*np.exp(E/N)
s = np.array([-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1])
CGF_ed = np.zeros(s.shape)
CGF_dmrg = np.zeros(s.shape)
all_energies = np.zeros((2**(N**2),len(s)))
for i in range(len(s)):
    if s[i] > -20 and s[i] < 0:
        target_state = 3
    else:
        target_state = 0
    x = mps_opt.MPS_OPT(N = [N,N],
                        hamType = "sep_2d",
                        periodic_x = True,
                        periodic_y = True,
                        target_state = target_state,
                        hamParams = (qx,px,1/2,1/2,0,0,0,0,0,0,0,0,[s[i]/N,0]))
    x.initialize_containers()
    x.generate_mpo()
    full_ham = x.mpo.return_block_ham()
    E_ed,_ = np.linalg.eig(full_ham)
    all_energies[:,i] = np.sort(E_ed)
    print(full_ham)
    print('{}'.format(np.sort(E_ed)))
    print('Energy via Exact Diagonalization: {}'.format(np.sort(E_ed)[-1]))
    print('Energy via Exact Diagonalization: {}'.format(np.sort(E_ed)[-3]))
    CGF_ed[i] = np.sort(E_ed)[-1]
    CGF_dmrg[i] = x.kernel()
"""

#-----------------------------------------------------------------------------
# 2D WASEP
#-----------------------------------------------------------------------------
N=4
n_points = 10
E = 10
px = 1/2*np.exp(-E/N)
qx = 1/2*np.exp(E/N)
s = np.array([-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1])
CGF_ed = np.zeros(s.shape)
CGF_dmrg = np.zeros(s.shape)
all_energies = np.zeros((2**(N**2),len(s)))
for i in range(len(s)):
    if s[i] > -20 and s[i] < 0:
        target_state = 3
    else:
        target_state = 0
    x = mps_opt.MPS_OPT(N = [N,N],
                        hamType = "sep_2d",
                        periodic_x = True,
                        periodic_y = True,
                        target_state = target_state,
                        hamParams = (qx,px,1/2,1/2,0,0,0,0,0,0,0,0,[s[i]/N,0]))
    x.initialize_containers()
    x.generate_mpo()
    for i in range(N**2):
        block_ham = x.mpo.return_single_block_ham(i)
        #print(block_ham)
        E_block,_ = np.linalg.eig(block_ham)
        print('Energies for Occupation Level {}'.format(i))
        print(np.sort(E_block))
    CGF_dmrg[i] = x.kernel()
