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
N=3
n_points = 10
E = 10
px = 1/2*np.exp(-E/N)
qx = 1/2*np.exp(E/N)
s = np.linspace(-30,10,n_points)
s = np.array([1,0.75,0.5,0.25,0.1,0.05,0,-0.05,-0.1,-0.15,-0.2,-0.25,-0.3,-0.4,-0.5,-0.6,-0.75,-1,-1.25,-1.5,-1.75,-2,-2.5,-3])
s = np.array([-3,-4,-5,-6,-7,-8,-9,-10])
s = np.array([-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21])
s = np.array([1,0.75,0.5,0.25,0.1,0,-0.01,-0.05,-0.1,-0.2,\
        -0.3,-0.4,-0.5,-0.75,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,\
        -11,-12,-13,-14,-15,-16,-17,-18,-19,-19.25,-19.5,\
        -19.6,-19.7,-19.8,-19.9,-19.95,-19.99,-20,-20.1,\
                -20.25,-20.5,-20.75,-21])
s = np.array([0])
CGF_ed = np.zeros(s.shape)
CGF_dmrg = np.zeros(s.shape)
all_energies = np.zeros((2**(N**2),len(s)))
for i in range(len(s)):
    x = mps_opt.MPS_OPT(N = [N,N],
                        hamType = "sep_2d",
                        periodic_x = True,
                        periodic_y = True,
                        hamParams = (qx,px,1/2,1/2,0,0,0,0,0,0,0,0,[s[i]/N,0]))
    x.initialize_containers()
    x.generate_mpo()
    full_ham = x.mpo.return_full_ham()
    E_ed,_ = np.linalg.eig(full_ham)
    all_energies[:,i] = np.sort(E_ed)
    print(full_ham)
    print('{}'.format(np.sort(E_ed)))
    print('Energy via Exact Diagonalization: {}'.format(np.sort(E_ed)[-1]))
    CGF_ed[i] = np.sort(E_ed)[-1]
    CGF_dmrg[i] = x.kernel()
