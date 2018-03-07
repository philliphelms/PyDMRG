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

Nx = 6
Ny = 6
x = mps_opt.MPS_OPT(N=Nx,
                    hamType="tasep",
                    plotExpVals=True,
                    plotConv=True,
                    maxBondDim=500,
                    add_noise=False,
                    hamParams=(0.35,-1,2/3))
x.kernel()
x1 = mps_opt.MPS_OPT(N=[Nx,Ny],
                    hamType="sep_2d",
                    plotExpVals=True,
                    plotConv=True,
                    maxBondDim = 500,
                    add_noise = False,
                    hamParams = (0,1,0.35,0,0,2/3,0,0,0,0,0,0,-1))
E1 = x1.kernel()
x2 = mps_opt.MPS_OPT(N=[Nx,Ny],
                     hamType="sep_2d",
                     plotExpVals=True,
                     plotConv=True,
                     maxBondDim = 500,
                     add_noise = False,
                     hamParams = (1,0,0,0.35,2/3,0,0,0,0,0,0,0,1))
E2 = x2.kernel()
x3 = mps_opt.MPS_OPT(N=[Ny,Nx],
                     hamType="sep_2d",
                     plotExpVals=True,
                     plotConv=True,
                     maxBondDim = 500,
                     add_noise = False,
                     hamParams = (0,0,0,0,0,0,1,0,0,0.35,2/3,0,1))
E3 = x3.kernel()
x4 = mps_opt.MPS_OPT(N=[Ny,Nx],
                     hamType="sep_2d",
                     plotExpVals=True,
                     plotConv=True,
                     maxBondDim = 500,
                     add_noise = False,
                     hamParams = (0,0,0,0,0,0,0,1,0.35,0,0,2/3,-1))
E4 = x4.kernel()
