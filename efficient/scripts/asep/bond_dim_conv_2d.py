import numpy as np
import time
import mps_opt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sys import argv

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
np.set_printoptions(precision=10)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

Nx = int(argv[1])
Ny = int(argv[2])
rho_r = 0.5
rho_l = 0.5
sx = np.linspace(-5,5,10)
px = np.array([0.1])
rho_right = 0.5
rho_left = 0.5
rho_top = 0.5
rho_bottom = 0.5
bd = np.array([2,4,6,8,10,20,30,40,50,60,70,80,90,100,125,150,175,200,250,300,350,400,450,500,600,700,800,900,1000])
print('sx =')
for i in range(len(sx)):
    print(sx[i])
print('px =')
for i in range(len(px)):
    print(px[i])
print('\n\n')
CGF = np.zeros((len(px),len(sx),len(bd)),dtype=np.complex128)   # CGF
nPart = np.zeros((len(px),len(sx),len(bd)),dtype=np.complex128) # Number of particles
EE = np.zeros((len(px),len(sx),len(bd)),dtype=np.complex128)    # Entanglement Entropy
density = np.zeros((len(px),len(sx),len(bd),Nx*Ny),dtype=np.complex128)
current = np.zeros((len(px),len(sx),len(bd),Nx*Ny),dtype=np.complex128)
CGF_ed = np.zeros((len(px),len(sx),len(bd)),dtype=np.complex128)    
nPart_ed = np.zeros((len(px),len(sx),len(bd)),dtype=np.complex128)
density_ed = np.zeros((len(px),len(sx),len(bd),Nx*Ny),dtype=np.complex128)
for k in range(len(bd)):
    for i in range(len(px)):
        for j in range(len(sx)):
            print('sx = {}'.format(sx[j]))
            print('px = {}'.format(px[i]))
            # Set up hopping rates
            jl = (1-px[i])*np.ones((Ny,Nx))
            jr = px[i]    *np.ones((Ny,Nx))
            jd = 0.5      *np.ones((Ny,Nx))
            ju = 0.5      *np.ones((Ny,Nx))
            cr = np.zeros((Ny,Nx))
            cr[:,0] = rho_left
            cl = np.zeros((Ny,Nx))
            cl[:,-1] = rho_right
            cu = np.zeros((Ny,Nx))
            cd = np.zeros((Ny,Nx))
            dr = np.zeros((Ny,Nx))
            dr[:,-1] = 1-rho_right
            dl = np.zeros((Ny,Nx))
            dl[:,0] = 1-rho_left
            du = np.zeros((Ny,Nx))
            dd = np.zeros((Ny,Nx))
            # Set up correct BCs
            periodic_y = True # Use PBC in y-direction
            x = mps_opt.MPS_OPT(N=[Nx,Ny],
                                maxBondDim = bd[k],
                                add_noise=False,
                                hamType = "sep_2d",
                                verbose = 2,
                                maxIter = 10,
                                tol = 1e-16,
                                hamParams = (jl,jr,jd,ju,cr,cl,cd,cu,dr,dl,dd,du,[sx[j],0.]))
            x.kernel()
            CGF[i,j,k] = x.finalEnergy
            EE[i,j,k] = x.entanglement_entropy[int(x.N/2)]
            nPart[i,j,k] = np.sum(x.calc_occ)
            density[i,j,k,:] = x.calc_occ
            current[i,j,k] = x.current
        np.savez('asep_2d_PBC_bond_dim_check_Nx'+str(Nx)+'_Ny'+str(Ny)+'_data_p'+str(len(px))+'s'+str(len(sx)),s=sx,p=px,CGF=CGF,EE=EE,nPart=nPart,density=density,current=current)
