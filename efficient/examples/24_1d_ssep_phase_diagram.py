import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Create a phase diagram by changing the input and output rates of the ssep
# in a 1D system. It may be useful to compare this to the 2D calculation shown
# in example 23.
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
npts = 10
betaVec = np.linspace(0,1,npts)
alphaVec = np.linspace(0,1,npts)
s_vec = np.array([-0.1,0.1])
J_mat = np.zeros((npts,npts))
for i in range(len(betaVec)):
    for j in range(len(alphaVec)):
        print('-'*20+'\nalpha = {}%, beta = {}%\n'.format(j/len(alphaVec)*100,i/len(betaVec)*100))
        print('alpha = {}, beta = {}\n'.format(alphaVec[j],betaVec[i])+'-'*20)
        x = mps_opt.MPS_OPT(N=N,
                            maxBondDim = [10,20,30],
                            hamType = "sep",
                            verbose = 0,
                            hamParams = (alphaVec[j],betaVec[i],0.5,0.5,betaVec[i],alphaVec[j],s_vec[0]))
        E1 = x.kernel()
        x = mps_opt.MPS_OPT(N=N,
                            maxBondDim = [10,20,30],
                            hamType = "sep",
                            verbose = 0,
                            hamParams = (alphaVec[j],betaVec[i],0.5,0.5,betaVec[i],alphaVec[j],s_vec[1]))
        E2 = x.kernel()
        J_mat[i,j] = np.abs((E1-E2)/(s_vec[1]-s_vec[0])/N)
x,y = np.meshgrid(betaVec,alphaVec)
f = plt.figure()
plt.pcolor(x,y,J_mat)
plt.colorbar()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('alpha',fontsize=20)
plt.ylabel('beta',fontsize=20)
f.savefig('my_dmrg_phaseDiagram_ssep_1D.pdf')
