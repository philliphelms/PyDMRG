import numpy as np
import mps_opt
import mps_opt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=10)
plt.style.use('ggplot')
colormap = cm.plasma

# Model Parameters
N = 10
Np = 1
Ns = 1000
rho_r = 0.5
rho_l = 0.5
if Np == 1:
    pVec = np.array([0.1])
else:
    pVec = np.linspace(0.,1.,Np)
sVec = np.linspace(-1.75,0,Ns)

CGF = np.zeros((len(pVec),len(sVec)),dtype=np.complex128)    
density = np.zeros((len(pVec),len(sVec),N),dtype=np.complex128)
rightEE = np.zeros((len(pVec),len(sVec)),dtype=np.complex128)

f1 = plt.figure()
f2 = plt.figure()
f3 = plt.figure()
for i,p in enumerate(pVec):
    for j,s in enumerate(sVec):
        print('s,p = {},{}'.format(s,p))
        if j == 0:
            mps_fname = 'rand'
        else:
            mps_fname = 'data/dmrg/tmpMPS.npz'
        x = mps_opt.MPS_OPT(N=N,
                            hamType = "sep",
                            maxBondDim=1000,
                            tol=1e-8,
                            mpsFilename='tmpMPS',
                            initialGuess=mps_fname,
                            hamParams = (rho_l,1-rho_l,p,1-p,1-rho_r,rho_r,s))
        CGF[i,j] = x.kernel()
        density[i,j,:] = x.calc_occ
        rightEE[i,j] = x.entanglement_entropy[int(N/2)]
        
        plt.figure(f1.number)
        plt.plot(sVec[:j],CGF[0,:j])
        plt.figure(f2.number)
        plt.plot(sVec[:j],density[0,:j,int(N/2)])
        plt.figure(f3.number)
        plt.plot(sVec[:j],rightEE[0,:j])
        plt.pause(0.00001)
plt.show()
