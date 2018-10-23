import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Run a simple calculation for a 2D Heisenberg Model
# DOESN'T WORK CORRECTLY!!!
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

x = mps_opt.MPS_OPT(N=[3,7],
                    hamType = "heis_2d",
                    verbose = 4,
                    plotExpVals = True,
                    plotConv = True,
                    periodic_x = True,
                    periodic_y = True,
                    maxBondDim=10,
                    hamParams = (1,0))
E = x.kernel()
