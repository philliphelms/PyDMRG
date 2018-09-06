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

N = int(argv[1])
rho_r = 0.5
rho_l = 0.5
s = np.linspace(-2,-0.5,15)
p = np.array([0.2])
bd = [2,4,6,8,10,20,30,40,50,60,70,80,90,100,125,150,175,200,250,300,350,400,450,500,600,700,800,900,1000]
maxIter = [2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2]
print('s =')
for i in range(len(s)):
    print(s[i])
print('p =')
for i in range(len(p)):
    print(p[i])
print('\n\n')
fname = str(int(time.time()))+'_'+'asep_bond_dim_check_N'+str(N)+'_data_p'+str(len(p))+'s'+str(len(s))+'bd'+str(len(bd))
CGF = np.zeros((len(p),len(s),len(bd)),dtype=np.complex128)   # CGF
nPart = np.zeros((len(p),len(s),len(bd)),dtype=np.complex128) # Number of particles
EE = np.zeros((len(p),len(s),len(bd)),dtype=np.complex128)    # Entanglement Entropy
density = np.zeros((len(p),len(s),len(bd),N),dtype=np.complex128)
current = np.zeros((len(p),len(s),len(bd),N),dtype=np.complex128)
for i in range(len(p)):
    for j in range(len(s)):
        print('s = {}'.format(s[j]))
        print('p = {}'.format(p[i]))
        x = mps_opt.MPS_OPT(N=N,
                            maxBondDim = bd,
                            maxIter = maxIter,
                            add_noise=False,
                            hamType = "sep",
                            verbose = 3,
                            hamParams = (rho_l,1-rho_l,p[i],1-p[i],1-rho_r,rho_r,s[j]))
        x.kernel()
        CGF[i,j,:] = x.bondDimEnergies
        EE[i,j,:] = x.bondDimEntanglement
        np.savez(fname,s=s,p=p,CGF=CGF,EE=EE)
