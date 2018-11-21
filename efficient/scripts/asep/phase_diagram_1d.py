import numpy as np
import time
import mps_opt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sys import argv
import time

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
s = np.linspace(-5,5,100)
s = np.array([-1.])
p = np.array([0.1])
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
current = np.zeros((len(p),len(s),N),dtype=np.complex128)
CGF_ed = np.zeros((len(p),len(s)),dtype=np.complex128)    
nPart_ed = np.zeros((len(p),len(s)),dtype=np.complex128)
density_ed = np.zeros((len(p),len(s),N),dtype=np.complex128)
for i in range(len(p)):
    for j in range(len(s)):
        print('s = {}'.format(s[j]))
        print('p = {}'.format(p[i]))
        x = mps_opt.MPS_OPT(N=N,
                            maxBondDim = 100,
                            add_noise=False,
                            hamType = "sep",
                            verbose = 4,
                            calc_psi=True,
#                            plotExpVals = True,
#                            plotConv = True,
                            hamParams = (rho_l,1-rho_l,p[i],1-p[i],1-rho_r,rho_r,s[j]))
        x.kernel()
        #CGF_ed[i,j] = x.exact_diag()
        #nPart_ed[i,j] = np.sum(x.ed.nv)
        CGF[i,j] = x.finalEnergy
        EE[i,j] = x.entanglement_entropy[int(x.N/2)]
        nPart[i,j] = np.sum(x.calc_occ)
        density[i,j,:] = x.calc_occ
        current[i,j] = x.current
        #density_ed[i,j,:] = x.ed.nv
        np.savez('N'+str(N)+'_data_p'+str(len(p))+'s'+str(len(s))+'Date'+str(time.time()),s=s,p=p,CGF=CGF,EE=EE,nPart=nPart,density=density,current=current)#CGF_ed=CGF_ed,nPart_ed=nPart_ed,density=density,density_ed=density_ed)
