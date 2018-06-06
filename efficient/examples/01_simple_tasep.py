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
np.set_printoptions(precision=3)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

# Create MPS object
x = mps_opt.MPS_OPT(N = 100,
                    hamType = 'tasep',
                    #plotExpVals = True,
                    maxBondDim = 100,
                    tol = 1e-5,
                    maxIter = 10,
                    verbose = 4,
                    periodic_x = False,
                    #plotConv = True,
                    #plotExpVals = True,
                    add_noise = False,
                    hamParams = (0.35,-1,2/3))
# Run optimization
x.kernel()
