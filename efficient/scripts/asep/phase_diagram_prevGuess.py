import numpy as np
import mps_opt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
from sys import argv
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

N = 10
Np = 1 
Ns = 1000 
rho_r = 0.5 
rho_l = 0.5 
if Np == 1:
    pVec = np.array([0.1])
else:
    pVec = np.linspace(0.,1.,Np)
sVec = np.linspace(-5,5,Ns)
mbdVec = np.array([2])

CGF = np.zeros((len(pVec),len(sVec),len(mbdVec)),dtype=np.complex128)    
density = np.zeros((len(pVec),len(sVec),len(mbdVec),N),dtype=np.complex128)
rightEE = np.zeros((len(pVec),len(sVec),len(mbdVec)),dtype=np.complex128)

for i,p in enumerate(pVec):
    for j,s in enumerate(sVec):
        for k,mbd in enumerate(mbdVec):
            print('s,p,mbd = {},{},{}'.format(s,p,mbd))
            if (j==0) and (i==0):
                x = mps_opt.MPS_OPT(N=N,
                                    hamType = "sep",
                                    maxBondDim = mbd,
                                    tol = 1e-8,
                                    maxIter = 5,
                                    mpsFilename = 'myMPS_'+str(mbd),
                                    hamParams = (rho_l,1-rho_l,p,1-p,1-rho_r,rho_r,s))
            else:
                print('Using Previous Guess')
                x = mps_opt.MPS_OPT(N=N,
                                    hamType = "sep",
                                    maxBondDim = mbd,
                                    tol = 1e-8,
                                    maxIter = 5,
                                    mpsFilename = 'myMPS_'+str(mbd),
                                    initialGuess = 'data/dmrg/myMPS_'+str(mbd)+'.npz',
                                    hamParams = (rho_l,1-rho_l,p,1-p,1-rho_r,rho_r,s))

            CGF[i,j,k] = x.kernel()
            density[i,j,k,:] = x.calc_occ
            rightEE[i,j,k] = x.entanglement_entropy[int(N/2)]
            fname = 'prevGuessPhaseDiagram_N'+str(N)+'_data_p'+str(Np)+'s'+str(Ns)
            np.savez(fname,
                     s=sVec,
                     p=pVec,
                     CGF=CGF,
                     density=density,
                     rightEE=rightEE)

f = plt.figure()
ax1 = f.add_subplot(131)
ax2 = f.add_subplot(132)
ax3 = f.add_subplot(133)
for mbdInd in range(len(mbdVec)):
    ax1.plot(sVec,CGF[0,:,mbdInd])
    ax2.plot(sVec,density[0,:,mbdInd,int(N/2)])
    ax3.plot(sVec,rightEE[0,:,mbdInd])
plt.show()
