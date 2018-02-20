import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Using the Ising Hamiltonian, perform a simple steady state calculation
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

N = 50
x = mps_opt.MPS_OPT(N=N,
                    hamType = "ising",
                    plotExpVals = True,
                    plotConv = True,
                    hamParams = (1,0))
E = x.kernel()
