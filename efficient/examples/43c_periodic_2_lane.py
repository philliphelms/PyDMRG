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

#########################################
# Calculation Parameters:
Nx = int(argv[1])
Ny = int(argv[2])
rho_right = 0.5
rho_left = 0.5
rho_top = 0.5
rho_bottom = 0.5
#px = np.linspace(0.,1.,10)
py = np.array([0.5])
sx = np.linspace(-2,-0.5,15)
sy = np.array([0])
px = np.array([0.2])
boundary_cond = 'periodic' #'periodic' 'open' 'closed'
#########################################
print('sx =')
for i in range(len(sx)):
    print(sx[i])
print('px =')
for i in range(len(px)):
    print(px[i])
print('sy =')
for i in range(len(sy)):
    print(sy[i])
print('py =')
for i in range(len(py)):
    print(py[i])
print('\n\n')
CGF = np.zeros((len(px),len(sx),len(px),len(sy)),dtype=np.complex128)   # CGF
nPart = np.zeros((len(px),len(sx),len(px),len(sy)),dtype=np.complex128) # Number of particles
EE = np.zeros((len(px),len(sx),len(px),len(sy)),dtype=np.complex128)    # Entanglement Entropy
density = np.zeros((len(px),len(sx),len(px),len(sy),Nx*Ny),dtype=np.complex128)
for i in range(len(px)):
    for j in range(len(sx)):
        for k in range(len(py)):
            for l in range(len(sy)):
                # Set up hopping rates
                jl = (1-px[i])*np.ones((Ny,Nx))
                jr = px[i]    *np.ones((Ny,Nx))
                jd = (1-py[k])*np.ones((Ny,Nx))
                ju = py[k]    *np.ones((Ny,Nx))
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
                if boundary_cond is 'periodic':
                    periodic_y = True
                elif boundary_cond is 'open':
                    cu[-1,:] = rho_bottom
                    cd[0,:] = rho_top
                    du[0,:] = 1-rho_top
                    dd[-1,:] = 1-rho_left
                    ju[0,:] = 0
                    jd[-1,:] = 0
                    periodic_y = False
                elif boundary_cond is 'closed':
                    ju[0,:] = 0
                    jd[-1,:] = 0
                    periodic_y = False
                # Set up and Run DMRG
                x = mps_opt.MPS_OPT(N=[Nx,Ny],
                                    add_noise=False,
                                    hamType = 'sep_2d',
                                    maxBondDim = 250,
                                    #plotExpVals = True,
                                    #plotConv = True,
                                    hamParams = (jl,jr,jd,ju,cr,cl,cd,cu,dr,dl,dd,du,[sx[j],sy[l]]))
                x.kernel()
                # Save Results
                CGF[i,j,k,l] = x.finalEnergy
                EE[i,j,k,l] = x.entanglement_entropy[int(x.N/2)]
                nPart[i,j,k,l] = np.sum(x.calc_occ)
                density[i,j,k,l,:] = x.calc_occ
                np.savez(str(time.time())+'_'+str(Nx)+'x'+str(Ny)+'_data_p'+str(len(px))+'_s'+str(len(py))+'pts_periodic',sx=sx,px=px,sy=sy,py=py,CGF=CGF,EE=EE,nPart=nPart,density=density)
print('CGF,EE,nPart')
for i in range(len(px)):
    for j in range(len(sx)):
        for k in range(len(py)):
            for l in range(len(sy)):
                print('{},{},{}'.format(np.real(CGF[i,j,k,l]),np.real(EE[i,j,k,l]),np.real(nPart[i,j,k,l])))
print('\n')
print('Density')
for i in range(len(px)):
    for j in range(len(sx)):
        for k in range(len(py)):
            for l in range(len(sy)):
                print(density[i,j,k,l,:])
