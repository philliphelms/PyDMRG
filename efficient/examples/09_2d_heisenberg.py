import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Run a simple calculation for a 2D Heisenberg Model
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
x = mps_opt.MPS_OPT(N=N**2,
                    hamType = "heis_2d",
                    plotExpVals = True,
                    plotConv = True,
                    maxBondDim=4,
                    hamParams = (1,0))
E = x.kernel()
