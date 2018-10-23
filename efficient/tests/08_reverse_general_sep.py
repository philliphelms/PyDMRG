import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# This calculation is identical to the one done in example 07, with the 
# exception that the current is now flowing in the opposite direction.
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

N = 10
x = mps_opt.MPS_OPT(N=N,
                    hamType = "sep",
                    plotExpVals = True,
                    plotConv = True,
                    periodic_x = False,
                    hamParams = (0,0.35,0,1,2/3,0,1))
E = x.kernel()
