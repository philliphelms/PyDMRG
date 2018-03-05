import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#----------------------------------------------------------
# A simple script to run a calculation for the tasep
# at a single point in phase space.
#----------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

# Create MPS object
x = mps_opt.MPS_OPT(N = 20,
                    hamType = 'tasep',
                    plotExpVals = True,
                    maxBondDim = [10,30,50],
                    tol = [1e-2,1e-5,1e-10],
                    maxIter = [2,5,10],
                    verbose = 3,
                    plotConv = True,
                    hamParams = (0.35,-1,2/3))
# Run optimization
x.kernel()
