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

N_vec = np.array([10,20,30,40,50])#,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200])
s = np.array([-1,1])
current = np.zeros(len(N_vec))
for i in range(len(N_vec)):
    N = int(N_vec[i])
    print('Running Calcs for N={}'.format(N))
    x = mps_opt.MPS_OPT(N=N,hamType='tasep',periodic_x=True,hamParams=(3/5,s[0],2/3))
    E_left = x.kernel()
    x = mps_opt.MPS_OPT(N=N,hamType='tasep',periodic_x=True,hamParams=(3/5,s[1],2/3))
    E_right = x.kernel()
    current[i] = (E_right-E_left)/(s[1]-s[0])/(N+1)
fig1 = plt.figure()
plt.plot(N_vec,-current,'ro-',markersize=5,linewidth=3)
plt.xlabel('$N$',fontsize=20)
plt.ylabel('$J(N)/(N+1)$',fontsize=20)
fig1.savefig('varySize.pdf')
