import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Here, we calculate the current and CGF as a function of s using both
# DMR and Mean Field Methods. We then compare the results and plot these.
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

# Recreate Ushnish plot
N = 8
s_vec = np.linspace(-2,2,20)
# E_dmrg = np.zeros(s_vec.shape)
E = np.zeros(s_vec.shape)
for i in range(len(s_vec)):
    x = mps_opt.MPS_OPT(N=N,
                        hamType = "sep",
                        hamParams = (0.9,0.1,0.5,0.5,0.9,0.1,s_vec[i]),
                        usePyscf = False)
    E[i] = x.mean_field()
fig1 = plt.figure()
plt.plot(s_vec,E,'-')
# plt.plot(s_vec,E_dmrg,':')
plt.grid(True)
fig1.savefig('vary_s_mf.pdf')
