import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Create a phase diagram for the tasep, showing current as a function of
# both alpha and beta. It is also compared to the phase diagram in the
# infinite limit and shows lines indicating the various phases.
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

N = 10
npts = 100
betaVec = np.linspace(0,1,npts)
alphaVec = np.linspace(0,1,npts)
J_mat = np.zeros((len(betaVec),len(alphaVec)))
J_mat_inf = np.zeros((len(betaVec),len(alphaVec)))
for i in range(len(betaVec)):
    for j in range(len(alphaVec)):
        print('-'*20+'\nalpha = {}%, beta = {}%\n'.format(j/len(alphaVec),i/len(betaVec)))
        x = mps_opt.MPS_OPT(N=int(N),
                            maxBondDim = 8,
                            tol = 1e-1,
                            verbose = 0,
                            hamParams = (alphaVec[j],-0.001,betaVec[i]))
        E1 = x.kernel()
        x = mps_opt.MPS_OPT(N=int(N),
                            maxBondDim = 8,
                            tol = 1e-1,
                            verbose = 0,
                            hamParams = (alphaVec[j],0.001,betaVec[i]))
        E2 = x.kernel()
        J_mat[i,j] = (E1-E2)/(0.002)/N
        # Determine infinite limit current
        if alphaVec[j] > 0.5 and betaVec[i] > 0.5:
            J_mat_inf[i,j] = 1/4
        elif alphaVec[j] < 0.5 and betaVec[i] > alphaVec[j]:
            J_mat_inf[i,j] = alphaVec[j]*(1-alphaVec[j])
        else:
            J_mat_inf[i,j] = betaVec[i]*(1-betaVec[i])
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
f2 = plt.figure()
plt.pcolor(x,y,J_mat_inf,vmin=-0,vmax=0.25)
plt.colorbar()
plt.plot(np.array([0,0.5]),np.array([0,0.5]),'k-',linewidth=5)
plt.plot(np.array([0.5,0.5]),np.array([0.5,1]),'k-',linewidth=5)
plt.plot(np.array([0.5,1]),np.array([0.5,0.5]),'k-',linewidth=5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('a',fontsize=20)
plt.ylabel('b',fontsize=20)
f2.savefig('analytic_phaseDiagram.pdf')
