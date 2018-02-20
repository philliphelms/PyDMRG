import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

# Possible Calculations #################################
bondDimScaling = True
sizeScaling = False
#########################################################

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'


if bondDimScaling:
    N = 100
    bondDimVec = np.array([2,4,6,8,10,12,14,16,18,20,30,40,50,60,70,80,90,100,110,120,130,140,150])
    times = np.zeros(bondDimVec.shape)
    for i in range(len(bondDimVec)):
        x = mps_opt.MPS_OPT(N = N,
                            maxIter = 1,
                            hamType = 'tasep',
                            hamParams = (0.35,-1,2/3),
                            maxBondDim = bondDimVec[i])
        t1 = time.time()
        x.kernel()
        t2 = time.time()
        times[i] = t2-t1
        print('Time for Bond Dim {} = {}'.format(bondDimVec[i],t2-t1))
    plt.figure()
    plt.loglog(bondDimVec,times)
    plt.show()

if sizeScaling:
    N = np.array([10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,240,280,320,360,400])
    times = np.array(N.shape)
    for i in range(len(bondDimVec)):
        x = mps_opt.MPS_OPT(N=N[i],
                            maxIter=1,
                            hamType='tasep',
                            hamParams=(0.35,-1,2/3),
                            maxBondDim = 20)
        t1 = time.time()
        x.kernel()
        t2 = time.time()
        times[i] = t2-t1
        print('Time for N = {} is {}'.format(N[i],t2-t1))
    plt.figure()
    plt.loglog(N,times)
    plt.show()
