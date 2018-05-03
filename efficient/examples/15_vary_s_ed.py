import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Here, we vary the value of s for a general SEP process. We compare the results
# for DMRG and Exact Diagonalization. 
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
s_vec = np.linspace(-2,2,10)
E_dmrg = np.zeros(s_vec.shape)
E = np.zeros(s_vec.shape)
for i in range(len(s_vec)):
    x = mps_opt.MPS_OPT(N=N,
                        hamType = "sep",
                        hamParams = (0.9,0.1,0.5,0.5,0.1,0.9,s_vec[i]),
                        #hamParams = (0.5,0.8,0.2,0.6,0.8,0.7,s_vec[i]),
                        usePyscf = True)
    E_dmrg[i] = x.kernel()
    E[i] = x.exact_diag()
fig1 = plt.figure()
plt.plot(s_vec,E,'-')
plt.plot(s_vec,E_dmrg,':')
plt.grid(True)
fig1.savefig('vary_s_ed.pdf')
