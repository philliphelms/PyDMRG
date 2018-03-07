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
Ny = Nx
n_points = 10
E = 10
px = 1/2*np.exp(-E/Nx)
qx = 1/2*np.exp(E/Nx)
s = np.linspace(0,5,n_points)
CGF = np.zeros(s.shape)
for i in range(len(s)):
    x1 = mps_opt.MPS_OPT(N=[Nx,Ny],
                        hamType="sep_2d",
                        plotExpVals=True,
                        plotConv=True,
                        maxBondDim = 100,#,300],#,500],
                        tol = 1e-3,#,1e-3],#,1e-4],
                        maxIter=3,#,3],#,10],
                        max_eig_iter=50,
                        verbose=4,
                        periodic_x=True,
                        periodic_y=True,
                        add_noise = False,
                        hamParams = (qx,px,0,0,0,0,0.5,0.5,0,0,0,0,[s[i],0]))
                        #(jump_left,jump_right,enter_left,enter_right,
                        # exit_left,exit_right,jump_up,jump_down,
                        # enter_top,enter_bottom,exit_top,exit_bottom,s))
    CGF[i] = x1.kernel()
