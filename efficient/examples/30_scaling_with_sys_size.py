import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Create a phase diagram by changing the input and output rates of the ssep
# in a 1D system. It may be useful to compare this to the 2D calculation shown
# in example 23.
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

# General Settings
maxBondDim = 500
maxIter = 3
max_eig_iter = 5
tol = 1e-16
verbose = 3
# Calculate scaling for 1D system
N = np.array([10,20,30,40,50,60,70,80,90,100,110,120,130,140])
t_vec = np.zeros(N.shape)
for i in range(len(N)):
    print('1D System: Size = {}'.format(N[i]))
    x = mps_opt.MPS_OPT(N=N[i],
                        maxBondDim = maxBondDim,
                        maxIter = maxIter,
                        max_eig_iter = max_eig_iter,
                        tol = tol,
                        verbose = verbose,
                        hamType = 'sep',
                        hamParams = (0.3,0.8,0.4,0.7,0.1,0.2,-1))
    t1 = time.clock()
    x.kernel()
    t2 = time.clock()
    t_vec[i] = t2-t1
    print('Computational Time = {}'.format(t_vec[i]))

# Calculate scaling for 2D system
N_vec_2d = np.array([2,4,6,8,10,12])#,14,16,18,20])
t_vec_2d = np.zeros(N_vec_2d.shape)
for i in range(len(N_vec_2d)):
    print('2D System: Size = {}'.format(N_vec_2d[i]**2))
    x = mps_opt.MPS_OPT(N=N_vec_2d[i]**2,
                        maxBondDim = maxBondDim,
                        maxIter = maxIter,
                        max_eig_iter = max_eig_iter,
                        tol = tol,
                        verbose = verbose,
                        hamType = 'sep_2d',
                        hamParams = (0.3,0.6,0.2,1.0,
                                     0.9,0.5,0.7,0.6,
                                     0.4,0.7,0.5,0.6,-1))
    t1 = time.clock()
    x.kernel()
    t2 = time.clock()
    t_vec_2d[i] = t2-t1
    print('Computational Time = {}'.format(t_vec_2d[i]))

fig1 = plt.figure()
plt.plot(N,t_vec,'ro')
plt.plot(N_vec_2d**2,t_vec_2d,'bo')
plt.xlabel('$N$')
plt.ylabel('Time')
plt.legend(('1D SEP','2D SEP'))
fig1.savefig('scaling_with_system_length.pdf')
