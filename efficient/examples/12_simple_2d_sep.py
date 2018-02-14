import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Perform a simple 2D SEP calculation
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

x = mps_opt.MPS_OPT(N=[10,4],
                    maxBondDim=500,
                    hamType="sep_2d",
                    plotExpVals=True,
                    plotConv=False,
                    verbose=3,
                    hamParams = (0.5,0.5,0.9,0.2,0.2,0.8,
                                 0.5,0.5,0.9,0.2,0.2,0.8,-1))
E = x.kernel()
