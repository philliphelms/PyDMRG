import numpy as np
import time
import mps_opt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sys import argv

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print('On rank {}'.format(rank))

#########################################
# Calculation Parameters:
Nx = int(argv[1])
Ny = int(argv[2])
nproc = int(argv[3])
rho_r = 0.5
rho_l = 0.5
sx = np.linspace(-2,-0.5,nproc-1)
px = np.array([0.2])
rho_right = 0.5
rho_left = 0.5
rho_top = 0.5
rho_bottom = 0.5
bd = [2,4,6,8,10,20,30,40,50,60,70,80,90,100,125,150,175,200,250,300,350,400,450,500,600,700,800,900,1000]
maxIter = [5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2]
fname = str(int(time.time()))+'_'+'asep_2d_PBC_bond_dim_check_Nx'+str(Nx)+'_Ny'+str(Ny)+'_data_p'+str(len(p))+'s'+str(len(s))+'bd'+str(len(bd))
#########################################
if rank == 0:
    CGF = np.zeros((len(px),len(sx),len(bd)),dtype=np.complex128)   # CGF
    EE = np.zeros((len(px),len(sx),len(bd)),dtype=np.complex128)    # Entanglement Entropy
    np.savez(fname,s=sx,p=px,bd=np.array(bd),maxIter=np.array(maxIter),CGF=CGF,EE=EE)
    for i in range(1, size):
        sendMsg = 'Save rank {}'.format(i)
        comm.send(sendMsg, dest=i)
        recMsg = comm.recv(source=i)
        print(recMsg)
else:
    i = 0
    j = rank - 1
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
    ju[0,:] = 0
    jd[-1,:] = 0
    periodic_y = False
    x = mps_opt.MPS_OPT(N=[Nx,Ny],
                        maxBondDim = bd,
                        add_noise=False,
                        hamType = "sep_2d",
                        maxIter = maxIter,
                        outputFile = '2d_sep_BD_check_Nx'+str(Nx)+'_Ny'+str(Ny)+str(int(time.time()))+'.log',
                        hamParams = (jl,jr,jd,ju,cr,cl,cd,cu,dr,dl,dd,du,[sx[j],0.]))
    x.kernel()
    recvMsg = comm.recv(source=0)
    print(recvMsg)
    print("Loading {}".format(fname))
    npzfile = np.load(fname+'.npz')
    print('Loaded {}'.format(fname))
    CGF = npzfile['CGF']
    EE = npzfile['EE']
    CGF[i,j,:] = x.bondDimEnergies
    EE[i,j,:] = x.bondDimEntanglement
    print('Saving {}'.format(fname))
    np.savez(fname,s=s,p=p,bd=np.array(bd),maxIter=np.array(maxIter),CGF=CGF,EE=EE)
    print('Saved {}'.format(fname))
    comm.send('Completed saving rank {}'.format(rank),dest=0)
    np.savez(fname,s=sx,p=px,bd=np.array(bd),maxIter=np.array(maxIter),CGF=CGF,EE=EE)
