import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# A simple calculation for the 1D heisenberg model.
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
x = mps_opt.MPS_OPT(N=int(N),
                    hamType = "heis",
                    add_noise=True,
                    plotExpVals = True,
                    plotConv = True,
                    periodic_x = True,
                    hamParams = (1,0))
E = x.kernel()
