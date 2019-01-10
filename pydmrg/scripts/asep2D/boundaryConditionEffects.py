from dmrg import *
from mpo.asep2D import return_mpo as return_mpo_asep2D
import time
from sys import argv
import os

# Set Calculation Parameters
Ny = int(argv[1])
Nx = 2
p = 0.1 
mbd = np.array([10])
bc = str(argv[2]) # Periodic, Open, Closed
sVec = np.linspace(-0.5,0.5,50)

# Allocate Memory for results
E   = np.zeros((len(sVec),len(mbd)))
EE  = np.zeros((len(sVec),len(mbd)))
gap = np.zeros((len(sVec),len(mbd)))
# Create spot to save MPS
id = str(int(time.time()))
path = 'mps/'+id+'/'
os.mkdir(path)
fname = path+'MPS_'+bc+'_Nx'+str(Nx)+'_Ny'+str(Ny)+'_id'

# Run Calculations
for sind,s in enumerate(sVec):
    if bc == 'periodic':
        mpo = return_mpo_asep2D((Nx,Ny), (0.5,  # jr
                                          0.5,  # jl
                                          p,    # ju
                                          1.-p, # jd
                                          0.,   # cr
                                          0.,   # cl
                                          0.5,  # cu
                                          0.5,  # cd
                                          0.,   # dr
                                          0.,   # dl
                                          0.5,  # du
                                          0.5,  # dd
                                          0.,   # sx
                                          s),   # xy
                                          periodicy = False,
                                          periodicx = True)
    elif bc == 'closed':
        mpo = return_mpo_asep2D((Nx,Ny), (0.5,  # jr
                                          0.5,  # jl
                                          p,    # ju
                                          1.-p, # jd
                                          0.,   # cr
                                          0.,   # cl
                                          0.5,  # cu
                                          0.5,  # cd
                                          0.,   # dr
                                          0.,   # dl
                                          0.5,  # du
                                          0.5,  # dd
                                          0.,   # sx
                                          s),   # xy
                                          periodicy=False,
                                          periodicx=False)
    elif bc == 'open':
        mpo = return_mpo_asep2D((Nx,Ny), (0.5,  # jr
                                          0.5,  # jl
                                          p,    # ju
                                          1.-p, # jd
                                          0.5,   # cr
                                          0.5,   # cl
                                          0.5,  # cu
                                          0.5,  # cd
                                          0.5,   # dr
                                          0.5,   # dl
                                          0.5,  # du
                                          0.5,  # dd
                                          0.,   # sx
                                          s),   # xy
                                          periodicy=False,
                                          periodicx=False)
    if sind == 0:
        E[sind,:],EE[sind,:],gap[sind,:] = run_dmrg(mpo,mbd=mbd,fname=fname,nStates=2,alg='exact')
    else:
        E[sind,:],EE[sind,:],gap[sind,:] = run_dmrg(mpo,mbd=mbd,initGuess=fname,fname=fname,nStates=2,alg='exact')
    # Save Results
    np.savez('results/asep2D_'+bc+'_psweep_Nx'+str(Nx)+'_Ny'+str(Ny)+'_Np1_Ns'+str(len(sVec)),N=(Nx,Ny),p=p,mbd=mbd,s=sVec,E=E,EE=EE,gap=gap)
