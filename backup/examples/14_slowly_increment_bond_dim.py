import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# For the default TASEP calculation, we compare the results of slowly 
# increasing the bond dimension to if we simply initially choose the
# largest bond dimension, to determine if it has any effect on final
# outcomes. And to compare time savings.
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

t1 = time.time()
N = 20
x = mps_opt.MPS_OPT(N=int(N),
                    verbose = 2,
                    maxBondDim = [10,50,100])
x.kernel()
t2 = time.time()
# Provide some comparison for if we don't slowly increase bond dim
x = mps_opt.MPS_OPT(N=int(N),
                    verbose = 2,
                    maxBondDim = 100)
x.kernel()
t3 = time.time()
# Print Results
print('#'*50+'\nIncremented case total time: {}\nDirect case total time: {}\n'.format(t2-t1,t3-t2)+'#'*50)
