from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print('On rank {}'.format(rank))

import numpy as np
import time
import mps_opt
from sys import argv

#########################################
# Calculation Parameters:
Nx = int(argv[1])
Ny = int(argv[2])
rho_right = 0.5
rho_left = 0.5
rho_top = 0.5
rho_bottom = 0.5
sx = np.linspace(-2,-0.5,15)
sy = np.array([0])
px = np.array([0.2])
py = np.array([0.5])
boundary_cond = 'periodic' 
#########################################
filename = str(int(time.time()))+'_'+str(Nx)+'x'+str(Ny)+'_data_p'+str(len(px))+'_s'+str(len(py))+'pts_periodic'
if rank == 0:
    CGF = np.zeros((len(px),len(sx),len(px),len(sy)),dtype=np.complex128)   # CGF
    nPart = np.zeros((len(px),len(sx),len(px),len(sy)),dtype=np.complex128) # Number of particles
    EE = np.zeros((len(px),len(sx),len(px),len(sy)),dtype=np.complex128)    # Entanglement Entropy
    density = np.zeros((len(px),len(sx),len(px),len(sy),Nx*Ny),dtype=np.complex128)
    np.savez(filename,sx=sx,px=px,sy=sy,py=py,CGF=CGF,EE=EE,nPart=nPart,density=density)
    print('Completed Rank {} Process'.format(rank))
    for i in range(1, size):
        sendMsg = 'Save rank {}'.format(i)
        comm.send(sendMsg, dest=i)
        recMsg = comm.recv(source=i)
        print(recMsg)
else:
    i = 0
    k = 0
    l = 0
    j = rank - 1
    print(j)
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
                        maxBondDim = 200,
                        periodic_y = periodic_y,
                        verbose = 3,
                        #plotExpVals = True,
                        #plotConv = True,
                        hamParams = (jl,jr,jd,ju,cr,cl,cd,cu,dr,dl,dd,du,[sx[j],sy[l]]))
    x.kernel()
    # Save Results
    recvMsg = comm.recv(source=0)
    print(recvMsg)
    print("Loading {}".format(filename))
    npzfile = np.load(filename+'.npz')
    print('Loaded {}'.format(filename))
    CGF = npzfile['CGF']
    EE = npzfile['EE']
    nPart = npzfile['nPart']
    density = npzfile['density']
    sx = npzfile['sx']
    sy = npzfile['sy']
    px = npzfile['px']
    py = npzfile['py']
    CGF[i,j,k,l] = x.finalEnergy
    EE[i,j,k,l] = x.entanglement_entropy[int(x.N/2)]
    nPart[i,j,k,l] = np.sum(x.calc_occ)
    density[i,j,k,l,:] = x.calc_occ
    print('Saving {}'.format(filename))
    np.savez(filename+'.npz',sx=sx,px=px,sy=sy,py=py,CGF=CGF,EE=EE,nPart=nPart,density=density)
    print('Saved {}'.format(filename))
    comm.send('Completed saving rank {}'.format(rank),dest=0)
