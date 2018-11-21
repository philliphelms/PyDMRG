import numpy as np
import time
import mps_opt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sys import argv
from matplotlib import cm
from calcEE import calcEE

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
Np = 100
Ns = 100
rho_r = 0.5
rho_l = 0.5
pVec = np.linspace(0.,1.,Np)
sVec = np.linspace(-5,5,Ns)

CGF_ed = np.zeros((len(pVec),len(sVec)),dtype=np.complex128)    
density_ed = np.zeros((len(pVec),len(sVec),N),dtype=np.complex128)
eigenSpec_ed = np.zeros((len(pVec),len(sVec),2**N),dtype=np.complex128)
rightEE_ed = np.zeros((len(pVec),len(sVec)),dtype=np.complex128)
leftEE_ed = np.zeros((len(pVec),len(sVec)),dtype=np.complex128)

for i,p in enumerate(pVec):
    for j,s in enumerate(sVec):
        print('s,p = {},{}'.format(s,p))
        x = mps_opt.MPS_OPT(N=N,
                            hamType = "sep",
                            hamParams = (rho_l,1-rho_l,p,1-p,1-rho_r,rho_r,s))
        CGF_ed[i,j] = x.exact_diag()
        density_ed[i,j,:] = x.ed.nv
        eigenSpec_ed[i,j,:] = x.ed.eigSpec[::-1]
        rightEE_ed[i,j] = calcEE(N,x.ed.rpsi,site=N/2)
        leftEE_ed[i,j] = calcEE(N,x.ed.lpsi,site=N/2)
        fname = 'FullPhaseDiagram_N'+str(N)+'_data_p'+str(Np)+'s'+str(Ns)
        np.savez(fname,
                 s=sVec,
                 p=pVec,
                 CGF_ed=CGF_ed,
                 density_ed=density_ed,
                 eigenSpec_ed = eigenSpec_ed,
                 rightEE_ed = rightEE_ed,
                 leftEE_ed = leftEE_ed)
