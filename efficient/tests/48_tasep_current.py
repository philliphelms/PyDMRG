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
np.set_printoptions(precision=10,linewidth=1000)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

N = 20
a = 0.35
b = 2/3
s = -1
ds = .001
# Create MPS object
x = mps_opt.MPS_OPT(N = N,
                    maxBondDim = 20,#[1,2,3,4,5,6,7,8,9,10,20,100],#[1,10,20,30,40,50,60,70,80,90,100],
                    hamType = 'tasep',
                    verbose = 4,
                    maxIter = 10,
                    leftMPS = True,
                    hamParams = (a,s,b))
x.kernel()
Current = x.current

# Compare to actual current
x1  = mps_opt.MPS_OPT(N = N,
                      maxBondDim = 100,
                      hamType = 'tasep',
                      verbose = 4,
                      hamParams = (a,s+ds,b))
E1 = x1.kernel()
x2  = mps_opt.MPS_OPT(N = N,
                      maxBondDim = 100,
                      hamType = 'tasep',
                      verbose = 4,
                      hamParams = (a,s-ds,b))
E2 = x2.kernel()
print('MPS Current = {}'.format(Current))
print('Derivative Current = {}'.format((E2-E1)/(2*ds)))

