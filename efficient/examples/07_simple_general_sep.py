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
np.set_printoptions(precision=1000)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

N = 100
x = mps_opt.MPS_OPT(N=N,
                    maxBondDim = 10,
                    hamType = "sep",
      #              plotExpVals = True,
      #              plotConv = True,
                    add_noise=False,
                    hamParams = (0.5,0.5,0.5,0.5,0.5,0.5,0))
                    #hamParams = (2/3,0,1,0,0,0.35,-1))
E = x.kernel()
#H = x.mpo.return_full_ham()
#print(H)
