import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# For the TASEP, we want to find the optimal spacing for s around s=0, to 
# determine the current. To do that, this runs calculations with many different
# spacings of ds to determine the optimal value.
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
ds = np.array([0.5,0.4,0.3,0.2,0.1,0.05,0.01,0.001,0.0001])
error = np.zeros(ds.shape)
betaVec = np.linspace(0.01,0.99,npts)
alphaVec = np.linspace(0.01,0.99,npts)
J_mat = np.zeros((len(betaVec),len(alphaVec)))
J_mat_inf = np.zeros((len(betaVec),len(alphaVec)))
J_mat_ed = np.zeros((len(betaVec),len(alphaVec)))
J_mat_mf = np.zeros((len(betaVec),len(alphaVec)))
for k in range(len(ds)):
    for i in range(len(betaVec)):
        for j in range(len(alphaVec)):
            print('-'*20+'\nalpha = {}% Complete\nbeta = {} Complete%\n'.format(j/len(alphaVec)*100,i/len(betaVec)*100))
            x = mps_opt.MPS_OPT(N=int(N),
                                hamParams = (alphaVec[j],-ds[k],betaVec[i]))
            E1 = x.kernel()
            print(E1)
            E1_ed = x.exact_diag()
            print(E1_ed)
            print(E1-E1_ed)
            x = mps_opt.MPS_OPT(N=int(N),
                                hamParams = (alphaVec[j],ds[k],betaVec[i]))
            E2 = x.kernel()
            print(E2)
            E2_ed = x.exact_diag()
            print(E2_ed)
            print(E2-E2_ed)
            # Calculate Current
            J_mat[i,j] = (E1-E2)/(2*ds[k])/N
            J_mat_ed[i,j] = (E1_ed-E2_ed)/(2*ds[k])/N
    error[k] = np.sum(np.sum(np.abs(J_mat-J_mat_ed)))/(len(alphaVec)*len(betaVec))
fig1 = plt.figure()
plt.semilogy(ds,np.abs(error))
fig1.savefig('test_ds.pdf')
