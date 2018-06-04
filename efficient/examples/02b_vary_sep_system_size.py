import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Calculations at a single point in phase space for the tasep, where we 
# increase the system size slowly and work towards the thermodynamic
# limit.
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

N_vec = np.array([100,50,100])#,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200])
s = np.array([-0.01,0.01])
current = np.zeros(len(N_vec))
for i in range(len(N_vec)):
    N = int(N_vec[i])
    print('Running Calcs for N={}'.format(N))
    x = mps_opt.MPS_OPT(N=N,
                        hamType='sep',
                        maxBondDim = 200,
                        verbose = 4,
                        plotConv = True,
                        hamParams = (0.5,0.5,0.2,0.8,0.8,0.5,-5))
    E_left = x.kernel()
