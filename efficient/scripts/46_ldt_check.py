import numpy as np
import time
import mps_opt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Here, I am going to calcualte some of the properties of the MPSs that we get
# and make sure that they are in agreement with those given by LDT. 
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=10)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

N = 2
rho_r = 0.5
rho_l = 0.5
p = 0.2
s = 0.1
x = mps_opt.MPS_OPT(N=N,
                    maxBondDim = 100,
                    add_noise=False,
                    hamType = "sep",
                    hamParams = (rho_l,1-rho_l,p,1-p,1-rho_r,rho_r,s))
x.kernel()
for i in range(len(x.rpsi)):
    print(x.rpsi[i])
print(np.sum(x.rpsi))
print(np.sum(x.rpsi*x.rpsi))
