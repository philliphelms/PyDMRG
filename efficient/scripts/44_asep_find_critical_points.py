import numpy as np
import time
import mps_opt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.optimize import fmin

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
p = np.linspace(0,0.5,20)[::-1]
sl = 0.
sr = 1000.
tol = 1e-5

s_min = np.zeros((len(p)))
cgf_min = np.zeros((len(p)))

for i in range(len(p)):
    print('Starting Opt for p = {}'.format(p[i]))
    def opt_func(x,*args):
            mps = mps_opt.MPS_OPT(N=N,maxBondDim=100,add_noise=False,hamType="sep",hamParams=(rho_l,1-rho_l,p[i],1-p[i],1-rho_r,rho_r,x),verbose=1)
            return mps.kernel()
    (s_min[i],cgf_min[i],_,_,_) = fmin(opt_func,s_min[i-1],full_output=True)
    print(s_min[i],cgf_min[i])
plt.figure()
plt.plot(p,s_min)
plt.show()
