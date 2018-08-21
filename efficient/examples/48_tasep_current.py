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

N = 10
a = 0.35
b = 2/3
s_vec = np.linspace(-5,5,100)
CGF = np.zeros(s_vec.shape)
Current = np.zeros(s_vec.shape)
# Create MPS object
for i in range(len(s_vec)):
    x = mps_opt.MPS_OPT(N = N,
                        maxBondDim = [10,20,100],
                        hamType = 'tasep',
                        verbose = 4,
                        hamParams = (a,s_vec[i],b))
    CGF[i] = x.kernel()
    Current[i] = x.current
plt.figure()
Ediff = CGF[1:]-CGF[:len(CGF)-1]
Sdiff = s_vec[1:]-s_vec[:len(s_vec)-1]
slope = -Ediff/(Sdiff)
plt.plot(s_vec[1:]+(s_vec[1]-s_vec[0])/2,slope,'-')
plt.plot(s_vec,Current,':')
plt.show()
