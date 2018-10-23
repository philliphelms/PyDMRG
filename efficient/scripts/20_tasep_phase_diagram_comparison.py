import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Compare the phase diagram for the TASEP as created via Exact Diagonalization,
# DMRG, and Mean Field
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
npts = 5
ds = 0.01
betaVec = np.linspace(0.01,0.99,npts)[::-1]
alphaVec = np.linspace(0.01,0.99,npts)[::-1]
J_mat = np.zeros((len(betaVec),len(alphaVec)))
J_mat_inf = np.zeros((len(betaVec),len(alphaVec)))
J_mat_ed = np.zeros((len(betaVec),len(alphaVec)))
J_mat_mf = np.zeros((len(betaVec),len(alphaVec)))
for i in range(len(betaVec)):
    for j in range(len(alphaVec)):
        print(('-'*20+'\nalpha = {}\nbeta = {}\n{}% Complete\n'+'-'*20).format(alphaVec[j],betaVec[i],i/len(betaVec)*100))
        x = mps_opt.MPS_OPT(N=int(N),
                            hamParams = (alphaVec[j],-ds,betaVec[i]))
        E1 = x.kernel()
        E1_ed = x.exact_diag()
        E1_mf = x.mean_field()
        x = mps_opt.MPS_OPT(N=int(N),
                            hamParams = (alphaVec[j],ds,betaVec[i]))
        E2 = x.kernel()
        E2_ed = x.exact_diag()
        E2_mf = x.mean_field()
        # Calculate Current
        J_mat[i,j] = (E1-E2)/(2*ds)/N
        J_mat_ed[i,j] = (E1_ed-E2_ed)/(2*ds)/N
        J_mat_mf[i,j] = (E1_mf-E2_mf)/(2*ds)/N
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
f3 = plt.figure()
plt.pcolor(x,y,J_mat_ed,vmin=-0,vmax=0.25)
plt.colorbar()
plt.plot(np.array([0,0.5]),np.array([0,0.5]),'k-',linewidth=5)
plt.plot(np.array([0.5,0.5]),np.array([0.5,1]),'k-',linewidth=5)
plt.plot(np.array([0.5,1]),np.array([0.5,0.5]),'k-',linewidth=5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('$\alpha$',fontsize=20)
plt.ylabel('$\beta$',fontsize=20)
f3.savefig('ed_phaseDiagram.pdf')
f4 = plt.figure()
plt.pcolor(x,y,J_mat_mf,vmin=-0,vmax=0.25)
plt.colorbar()
plt.plot(np.array([0,0.5]),np.array([0,0.5]),'k-',linewidth=5)
plt.plot(np.array([0.5,0.5]),np.array([0.5,1]),'k-',linewidth=5)
plt.plot(np.array([0.5,1]),np.array([0.5,0.5]),'k-',linewidth=5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('$\alpha$',fontsize=20)
plt.ylabel('$\beta$',fontsize=20)
f4.savefig('mf_phaseDiagram.pdf')
f5 = plt.figure()
plt.pcolor(x,y,np.log(np.abs(J_mat_ed-J_mat+1e-16)))
plt.colorbar()
plt.plot(np.array([0,0.5]),np.array([0,0.5]),'k-',linewidth=5)
plt.plot(np.array([0.5,0.5]),np.array([0.5,1]),'k-',linewidth=5)
plt.plot(np.array([0.5,1]),np.array([0.5,0.5]),'k-',linewidth=5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('$\alpha$',fontsize=20)
plt.ylabel('$\beta$',fontsize=20)
f5.savefig('dmrg_phaseDiagram_error.pdf')
f6 = plt.figure()
plt.pcolor(x,y,np.log(np.abs(J_mat_ed-J_mat_mf+1e-16)))
plt.colorbar()
plt.plot(np.array([0,0.5]),np.array([0,0.5]),'k-',linewidth=5)
plt.plot(np.array([0.5,0.5]),np.array([0.5,1]),'k-',linewidth=5)
plt.plot(np.array([0.5,1]),np.array([0.5,0.5]),'k-',linewidth=5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('$\alpha$',fontsize=20)
plt.ylabel('$\beta$',fontsize=20)
f6.savefig('mf_phaseDiagram_error.pdf')
