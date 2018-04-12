import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# A simple calculation using the general sep instead of the tasep. This
# is initially set up to run the case identical to the one done in the 
# 01_simple_tasep.py example.
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
a = 2/3
g = 0
p = 1
q = 0
b = 0
d = 0.35
s = -1
# Check non periodic case
x = mps_opt.MPS_OPT(N=N,
                    maxBondDim = 20,
                    hamType = "sep",
                    plotExpVals = True,
                    plotConv = True,
                    add_noise=False,
                    hamParams = (a,g,p,q,b,d,s))
E = x.kernel()
# Start playing with vector input
a_vec = np.zeros(N)
a_vec[0] = a
g_vec = np.zeros(N)
p_vec = np.ones(N)
q_vec = np.zeros(N)
b_vec = np.zeros(N)
d_vec = np.zeros(N)
d_vec[-1] = d
x = mps_opt.MPS_OPT(N=N,
                    maxBondDim = 20,
                    hamType = "sep",
                    plotExpVals = True,
                    plotConv = True,
#                    periodic_x = True,
                    add_noise=False,
                    hamParams = (a_vec,g_vec,p_vec,q_vec,b_vec,d_vec,s))
E = x.kernel()
# Start playing with PBC
a_vec = np.zeros(N)
a_vec[0] = a
g_vec = np.zeros(N)
p_vec = np.ones(N)
p_vec[-1] = 0
q_vec = np.zeros(N)
b_vec = np.zeros(N)
d_vec = np.zeros(N)
d_vec[-1] = d
x = mps_opt.MPS_OPT(N=N,
                    maxBondDim = 20,
                    hamType = "sep",
                    plotExpVals = True,
                    plotConv = True,
                    periodic_x = True,
                    add_noise=False,
                    hamParams = (a_vec,g_vec,p_vec,q_vec,b_vec,d_vec,s))
E = x.kernel()
# Continue playing with PBC
dividing_point = 0
a_vec = np.zeros(N)
a_vec[dividing_point] = a
g_vec = np.zeros(N)
p_vec = np.ones(N)
p_vec[dividing_point-1] = 0
q_vec = np.zeros(N)
b_vec = np.zeros(N)
d_vec = np.zeros(N)
d_vec[dividing_point-1] = d
x = mps_opt.MPS_OPT(N=N,
                    maxBondDim = 20,
                    hamType = "sep",
                    plotExpVals = True,
                    plotConv = True,
                    periodic_x = True,
                    add_noise=False,
                    hamParams = (a_vec,g_vec,p_vec,q_vec,b_vec,d_vec,s))
E = x.kernel()
# Try Backwards now
b = a
g = d
a = 0
p = 0
q = 1
d = 0
s = -s
# Check non periodic case
x = mps_opt.MPS_OPT(N=N,
                    maxBondDim = 20,
                    hamType = "sep",
                    plotExpVals = True,
                    plotConv = True,
                    add_noise=False,
                    hamParams = (a,g,p,q,b,d,s))
E = x.kernel()
# Start playing with vector input
a_vec = np.zeros(N)
g_vec = np.zeros(N)
g_vec[0] = g
p_vec = np.zeros(N)
q_vec = np.ones(N)
b_vec = np.zeros(N)
b_vec[-1] = b
d_vec = np.zeros(N)
x = mps_opt.MPS_OPT(N=N,
                    maxBondDim = 20,
                    hamType = "sep",
                    plotExpVals = True,
                    plotConv = True,
#                    periodic_x = True,
                    add_noise=False,
                    hamParams = (a_vec,g_vec,p_vec,q_vec,b_vec,d_vec,s))
E = x.kernel()
# Start playing with PBC
a_vec = np.zeros(N)
g_vec = np.zeros(N)
g_vec[0] = g
p_vec = np.zeros(N)
q_vec = np.ones(N)
q_vec[0] = 0
b_vec = np.zeros(N)
b_vec[-1] = b
d_vec = np.zeros(N)
x = mps_opt.MPS_OPT(N=N,
                    maxBondDim = 20,
                    hamType = "sep",
                    plotExpVals = True,
                    plotConv = True,
                    periodic_x = True,
                    add_noise=False,
                    hamParams = (a_vec,g_vec,p_vec,q_vec,b_vec,d_vec,s))
E = x.kernel()
# Continue playing with PBC
dividing_point = 5
a_vec = np.zeros(N)
g_vec = np.zeros(N)
g_vec[dividing_point] = g
p_vec = np.zeros(N)
q_vec = np.ones(N)
q_vec[dividing_point] = 0
b_vec = np.zeros(N)
b_vec[dividing_point-1] = b
d_vec = np.zeros(N)
x = mps_opt.MPS_OPT(N=N,
                    maxBondDim = 20,
                    hamType = "sep",
                    plotExpVals = True,
                    plotConv = True,
                    periodic_x = True,
                    add_noise=False,
                    hamParams = (a_vec,g_vec,p_vec,q_vec,b_vec,d_vec,s))
E = x.kernel()
