import numpy as np
import time
import mps_opt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

N = 6
rho_r = 0.5
rho_l = 0.5
p = np.linspace(-1,1,20)
s = np.linspace(-1,1,20)
CGF = np.zeros((len(p),len(s)))   # CGF
nPart = np.zeros((len(p),len(s))) # Number of particles
EE = np.zeros((len(p),len(s)))    # Entanglement Entropy
for i in range(len(p)):
    print(i)
    for j in range(len(s)):
        print('\t{}'.format(j))
        x = mps_opt.MPS_OPT(N=N,
                            add_noise=False,
                            hamType = "sep",
                            hamParams = (rho_l,1-rho_l,p[i],1-p[i],1-rho_r,rho_r,s[j]))
        x.kernel()
        print(x.final_convergence)
        print(x.final_convergence)
        if x.final_convergence == True:
            CGF[i,j] = x.finalEnergy
            EE[i,j] = x.entanglement_entropy[int(N/2)]
            nPart[i,j] = np.sum(x.calc_occ)
print(CGF)
print(EE)
print(nPart)
p,s = np.meshgrid(p,s)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(p,s,CGF)
plt.show()
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(p,s,EE)
plt.show()
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(p,s,nPart)
plt.show()

