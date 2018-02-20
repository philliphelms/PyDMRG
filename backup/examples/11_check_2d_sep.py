import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Ensure that the 2D sep calculations are correct by doing a tasep calculation
# aligned in all four possible directions. Compare the results to ensure the 
# resulting energies are coincident.
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

N = 4
x1 = mps_opt.MPS_OPT(N=N**2,
                    hamType="sep_2d",
                    plotExpVals=True,
                    plotConv=True,
                    hamParams = (0,1,0.35,0,0,2/3,      # jl,jr,il,ir,ol,or,
                                 0,0,0,   0,0,0  ,0))  # ju,jd,it,ib,ot,ob,s
E1 = x1.kernel()
x2 = mps_opt.MPS_OPT(N=N**2,
                     hamType="sep_2d",
                     plotExpVals=True,
                     plotConv=True,
                     hamParams = (1,0,0,0.35,2/3,0,
                                  0,0,0,0   ,0  ,0,0))
E2 = x2.kernel()
x3 = mps_opt.MPS_OPT(N=N**2,
                     hamType="sep_2d",
                     plotExpVals=True,
                     plotConv=True,
                     hamParams = (0,0,0,0,0,0,
                                  1,0,0,0.35,2/3,0,0))
E3 = x3.kernel()
x4 = mps_opt.MPS_OPT(N=N**2,
                     hamType="sep_2d",
                     plotExpVals=True,
                     plotConv=True,
                     hamParams = (0,0,0,0,0,0,
                                  0,1,0.35,0,0,2/3,0))
E4 = x4.kernel()
