import numpy as np
import time
import mps_opt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sys import argv
from matplotlib import cm

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=16)
np.set_printoptions(linewidth=1000)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'
plt.style.use('seaborn-bright') #'fivethirtyeight'
colormap = cm.plasma #coolwarm, inferno, viridis

N = 10
rho_r = 0.5
rho_l = 0.5
Ns = 50
sVec = np.linspace(-0.3,0.3,Ns)
pVec = np.array([0.1])
bdVec = [1,2,4,6,8,10,16,24,32,64]#,20,30,40,50]#,60,70,80,90,100,125,150,175,200,250,300,350,400,450,500,600,700,800,900,1000]
maxIterVec = [5,5,5,5,5,5,5,5,5,5]#,10,10,10,10]#[2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2]

print('s =')
for i in range(len(sVec)):
    print(sVec[i])
print('p =')
for i in range(len(pVec)):
    print(pVec[i])
print('\n\n')


fname = str(int(time.time()))+'_'+'asep_bond_dim_check_N'+str(N)+'_data_p'+str(len(pVec))+'_s'+str(len(sVec))+'_bd'+str(len(bdVec))
CGF = np.zeros((len(pVec),len(sVec),len(bdVec)),dtype=np.complex128)   # CGF
EE = np.zeros((len(pVec),len(sVec),len(bdVec)),dtype=np.complex128)    # Entanglement Entropy
EEs = np.zeros((len(pVec),len(sVec),len(bdVec),max(bdVec)),dtype=np.complex128)   # Entanglement Spectrum


fig = plt.figure()
ax = fig.gca()
for pind,p in enumerate(pVec):
    for sind,s in enumerate(sVec):
        print('s = {}'.format(s))
        print('p = {}'.format(p))
        x = mps_opt.MPS_OPT(N=N,
                            maxBondDim = bdVec,
                            maxIter = maxIterVec,
                            add_noise=False,
                            hamType = "sep",
                            verbose = 4,
                            hamParams = (rho_l,1-rho_l,p,1-p,1-rho_r,rho_r,s))
        x.kernel()
        CGF[pind,sind,:] = x.bondDimEnergies
        EE[pind,sind,:] = x.bondDimEntanglement
        EEs[pind,sind,:,:] = x.bondDimEntanglementSpec
        np.savez(fname,s=sVec,p=pVec,bd=bdVec,CGF=CGF,EE=EE,EEs=EEs)
        ax.clear()
        for bd_ind in range(len(bdVec)):
            #ax.semilogy(sVec[:sind+1],np.abs((CGF[0,:sind+1,bd_ind]-CGF[0,:sind+1,-1])/CGF[0,:sind+1,-1]),'o-',label='$M='+str(bdVec[bd_ind])+'$',color=colormap(bd_ind/len(bdVec)))
            ax.semilogy(sVec[:sind+1],np.abs((CGF[0,:sind+1,bd_ind]-CGF[0,:sind+1,-1])),'o-',label='$M='+str(bdVec[bd_ind])+'$',color=colormap(bd_ind/len(bdVec)))
        ax.legend()
        plt.pause(0.01)

