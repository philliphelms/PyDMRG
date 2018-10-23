import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Create a phase diagram for the ssep, showing current as a function of
# both alpha and beta.
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

N = 4
npts = 20
betaVec = np.linspace(0,1,npts)
alphaVec = np.linspace(0,1,npts)
J_mat = np.zeros((len(betaVec),len(alphaVec)))
for i in range(len(betaVec)):
    for j in range(len(alphaVec)):
        print('-'*20+'\nalpha = {}%, beta = {}%\n'.format(j/len(alphaVec),i/len(betaVec)))
        x = mps_opt.MPS_OPT(N=int(N),
                            maxBondDim = 10,
                            tol = 1e-5,
                            verbose = 0,
                            hamType = 'sep',
                            periodic_x = True,
                            hamParams = (0,0,alphaVec[j],betaVec[i],0,0,-0.001))
        E1 = x.kernel()
        x = mps_opt.MPS_OPT(N=int(N),
                            maxBondDim = 10,
                            tol = 1e-1,
                            verbose = 0,
                            periodic_x = True,
                            hamParams = (0,0,alphaVec[j],betaVec[i],0,0,0.001))
        E2 = x.kernel()
        J_mat[i,j] = (E1-E2)/(0.002)/N
x,y = np.meshgrid(betaVec,alphaVec)
f = plt.figure()
plt.pcolor(x,y,J_mat,vmin=-0,vmax=0.25)
plt.colorbar()
plt.plot(np.array([0,0.5]),np.array([0,0.5]),'k-',linewidth=5)
plt.plot(np.array([0.5,0.5]),np.array([0.5,1]),'k-',linewidth=5)
plt.plot(np.array([0.5,1]),np.array([0.5,0.5]),'k-',linewidth=5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('$\alpha$',fontsize=20)
plt.ylabel('$\beta$',fontsize=20)
f.savefig('dmrg_phaseDiagram.pdf')
