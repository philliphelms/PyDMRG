import numpy as np
import time
import mps_opt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sys import argv

#-----------------------------------------------------------------------------
# A simple calculation using the general sep instead of the tasep. This
# is initially set up to run the case identical to the one done in the 
# 01_simple_tasep.py example.
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=10)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

N = int(argv[2]) 
rho_r = 0.5
rho_l = 0.5
p = np.linspace(0.,1.,int(argv[1]))
p = np.array([0.1])
s = np.linspace(-5,1,int(argv[1]))
filename='asep_growing_sys_10x10_data_spacing'+str(int(argv[1]))+'_N'+str(int(argv[2]))
print('s =')
for i in range(len(s)):
    print(s[i])
print('p =')
for i in range(len(p)):
    print(p[i])
print('\n\n')
CGF = np.zeros((len(p),len(s)),dtype=np.complex128)   # CGF
nPart = np.zeros((len(p),len(s)),dtype=np.complex128) # Number of particles
EE = np.zeros((len(p),len(s)),dtype=np.complex128)    # Entanglement Entropy
density = np.zeros((len(p),len(s),N),dtype=np.complex128)
CGF_ed = np.zeros((len(p),len(s)),dtype=np.complex128)    # Entanglement Entropy
nPart_ed = np.zeros((len(p),len(s)),dtype=np.complex128)    # Entanglement Entropy
density_ed = np.zeros((len(p),len(s),N),dtype=np.complex128)

for i in range(len(p)):
    for j in range(len(s)):
        print('s = {}'.format(s[j]))
        print('p = {}'.format(p[i]))
        x = mps_opt.MPS_OPT(N=N,
                            maxBondDim = 200,
                            add_noise=False,
                            hamType = "sep",
                            verbose = 2,
                            #plotExpVals = True,
                            #plotConv = True,
                            hamParams = (rho_l,1-rho_l,p[i],1-p[i],1-rho_r,rho_r,s[j]))
        x.kernel()
        CGF_ed[i,j] = x.exact_diag()
        nPart_ed[i,j] = np.sum(x.ed.nv)
        CGF[i,j] = x.finalEnergy
        EE[i,j] = x.entanglement_entropy[int(x.N/2)]
        nPart[i,j] = np.sum(x.calc_occ)
        density[i,j,:] = x.calc_occ
        density_ed[i,j,:] = x.ed.nv
    np.savez(filename, CGF=CGF,CGF_ed=CGF_ed,nPart_ed=nPart_ed,EE=EE,density=density,density_ed=density_ed)
    print('CGF for p = {}'.format(p[i]))
    for j in range(len(s)):
        print(np.real(CGF[i,j]))
    print('EE for p = {}'.format(p[i]))
    for j in range(len(s)):
        print(np.real(EE[i,j]))
    print('nPart for p = {}'.format(p[i]))
    for j in range(len(s)):
        print(np.real(nPart[i,j]))
    print('CGF (ed) for p = {}'.format(p[i]))
    for j in range(len(s)):
        print(np.real(CGF_ed[i,j]))
    print('nPart (ed) for p = {}'.format(p[i]))
    for j in range(len(s)):
        print(np.real(nPart_ed[i,j]))
