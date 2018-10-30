import numpy as np
import time
import mps_opt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=10)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

N = 10
rho_r = 0.5
rho_l = 0.5
p = 0.2
s = -3.
diff = np.array([1e-1,1e-2,1e-3,1e-4,1e-5,1e-6])

x = mps_opt.MPS_OPT(N=N,
                    maxBondDim = 100,
                    add_noise=False,
                    hamType = "sep",
                    hamParams = (rho_l,1-rho_l,p,1-p,1-rho_r,rho_r,s))
x.kernel()
exact_current = x.current

curr = np.zeros(len(diff))
for i in range(len(diff)):
    x1 = mps_opt.MPS_OPT(N=N,
                        maxBondDim = 100,
                        add_noise=False,
                        hamType = "sep",
                        hamParams = (rho_l,1-rho_l,p,1-p,1-rho_r,rho_r,s-diff[i]))
    left_cgf = x1.kernel()
    x2 = mps_opt.MPS_OPT(N=N,
                        maxBondDim = 100,
                        add_noise=False,
                        hamType = "sep",
                        hamParams = (rho_l,1-rho_l,p,1-p,1-rho_r,rho_r,s+diff[i]))
    right_cgf = x2.kernel()
    curr[i] = (right_cgf-left_cgf)/(2*diff[i])
    print(curr[i])
plt.figure()
plt.plot(np.array([diff[0],diff[-1]]),np.array([exact_current,exact_current]))
plt.plot(diff,curr)
plt.show()
