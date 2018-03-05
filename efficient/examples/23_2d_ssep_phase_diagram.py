import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Create a phase diagram by varying alpha and beta for the ssep process for 
# a two dimensional system with closed edges on the top and bottom of the 
# system.
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

N = 8
npts = 30
betaVec = np.linspace(0,1,npts)
alphaVec = np.linspace(0,1,npts)
s_vec = np.array([-0.1,0.1])
J_mat = np.zeros((npts,npts))
for i in range(len(betaVec)):
    for j in range(len(alphaVec)):
        print('-'*20+'\nalpha = {}%, beta = {}%\n'.format(j/len(alphaVec)*100,i/len(betaVec)*100))
        print('alpha = {}, beta = {}\n'.format(alphaVec[j],betaVec[i])+'-'*20)
        x = mps_opt.MPS_OPT(N=[N,N],
                            maxBondDim = [10,30,50],
                            hamType = "sep_2d",
                            verbose = 2,
                            hamParams = (0.25,0.25,0,0,0,0,
                                         0.25,0.25,alphaVec[j],betaVec[i],betaVec[i],alphaVec[j],s_vec[0]))
        E1 = x.kernel()
        x = mps_opt.MPS_OPT(N=[N,N],
                            maxBondDim = [10,30,50],
                            hamType = "sep_2d",
                            verbose = 2,
                            hamParams = (0.25,0.25,0,0,0,0,
                                         0.25,0.25,alphaVec[j],betaVec[i],betaVec[i],alphaVec[j],s_vec[1]))
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
f.savefig('my_dmrg_phaseDiagram_ssep_2d.pdf')
