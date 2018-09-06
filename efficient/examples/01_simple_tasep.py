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
N = 30
a = 0.35
b = 2/3
s = -1
ds = 0.01
x = mps_opt.MPS_OPT(N = N,
                    hamType = 'tasep',
                    verbose = 4,
                    leftMPS = False,
                    hamParams = (a,s,b))
# Run optimization
x.kernel()
# Calculate current
print('Calculated Current: {}'.format(x.current))

# Use derivative of CGF
x = mps_opt.MPS_OPT(N=N,hamType='tasep',hamParams=(a,s+ds,b))
E1 = x.kernel()
x = mps_opt.MPS_OPT(N=N,hamType='tasep',hamParams=(a,s-ds,b))
E2 = x.kernel()
print('Derivative of CGF : {}'.format((E1-E2)/(2*ds)))
