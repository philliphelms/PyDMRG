import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt


# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot')

def run_test():
    N_vec = np.array([10,15,20])
    s = np.array([-0.01,0.01])
    current = np.zeros(len(N_vec))
    for i in range(len(N_vec)):
        N = int(N_vec[i])
        x1 = mps_opt.MPS_OPT(N=N,
                            hamType='sep',
                            maxBondDim = 50,
                            periodic_x=False,
                            add_noise = False,
                            tol = 1e-3,
                            verbose = 0,
                            hamParams=(0.5,0.5,0.2,0.8,0.8,0.5,s[0]))
        E_left = x1.kernel()
        x2 = mps_opt.MPS_OPT(N=N,
                            hamType='sep',
                            maxBondDim = 50,
                            periodic_x=False,
                            add_noise = False,
                            tol = 1e-3,
                            verbose = 0,
                            hamParams=(0.5,0.5,0.2,0.8,0.8,0.5,s[1]))
        E_right = x2.kernel()
        current[i] = (E_right-E_left)/(s[1]-s[0])/(N+1)
    fig1 = plt.figure()
    plt.plot(N_vec,-current,'ro-',markersize=5,linewidth=3)
    plt.xlabel('$N$',fontsize=20)
    plt.ylabel('$J(N)/(N+1)$',fontsize=20)
    fig1.savefig('results/test/varySepSize.pdf')
    return current
