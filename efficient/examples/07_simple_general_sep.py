import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# A simple calculation using the general sep instead of the tasep. This
# is initially set up to run the case identical to the one done in the 
# 01_simple_tasep.py example.
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

N = 30
x = mps_opt.MPS_OPT(N=N,
                    maxBondDim = 200,
                    hamType = "sep",
                    plotExpVals = True,
                    plotConv = True,
                    periodic_x = True,
                    add_noise=False,
                    hamParams = (2/3,0,1,0,0,0.35,-1))
E = x.kernel()
