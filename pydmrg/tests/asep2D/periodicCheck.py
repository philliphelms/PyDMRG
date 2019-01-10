from dmrg import *
from mpo.asep import return_mpo as return_mpo_asep
from mpo.asep2D import return_mpo as return_mpo_asep2D
import time

# Run a check to ensure PBC are working by setting up staggered
# SEPs on a 2D lattice in both directions
def run_test():
    N = 3
    mbd = 10
    hamParams = (np.random.rand(), # in on left
                 np.random.rand(), # Out on left
                 np.random.rand(), # forward hop
                 np.random.rand(), # backward hop
                 np.random.rand(), # Out on right
                 np.random.rand(), # In on right
                 np.random.rand()) # Bias


    # 1D For comparison
    mpo = return_mpo_asep(N,hamParams)
    E1,_,_ = run_dmrg(mpo,mbd=mbd,nStates=1)

    # Simple Vertically Stacked 2D
    Nx = N
    Ny = N
    mpo = return_mpo_asep2D(N,(hamParams[2],
                               hamParams[3],
                               0,
                               0,
                               hamParams[0],
                               hamParams[5],
                               0,
                               0,
                               hamParams[4],
                               hamParams[1],
                               0,
                               0,
                               hamParams[6],
                               0))
    E2,_,_ = run_dmrg(mpo,mbd=mbd,nStates=1,alg='arnoldi')
    E2 = E2/N

    # Simple Backwards Vertically Stacked 2D
    Nx = N
    Ny = N
    mpo = return_mpo_asep2D(N,(hamParams[3],
                               hamParams[2],
                               0,
                               0,
                               hamParams[5],
                               hamParams[0],
                               0,
                               0,
                               hamParams[1],
                               hamParams[4],
                               0,
                               0,
                               -hamParams[6],
                               0))
    E3,_,_ = run_dmrg(mpo,mbd=mbd,nStates=1,alg='arnoldi')
    E3 = E3/N

    # Simple Horizontally Stacked 2D?
    mpo = return_mpo_asep2D(N,(0,
                               0,
                               hamParams[2],
                               hamParams[3],
                               0,
                               0,
                               hamParams[0],
                               hamParams[5],
                               0,
                               0,
                               hamParams[4],
                               hamParams[1],
                               0,
                               hamParams[6]))
    E4,_,_ = run_dmrg(mpo,mbd=mbd,nStates=1,alg='arnoldi')
    E4 = E4/N

    # Simple Horizontally Stacked 2D?
    mpo = return_mpo_asep2D(N,(0,
                               0,
                               hamParams[3],
                               hamParams[2],
                               0,
                               0,
                               hamParams[5],
                               hamParams[0],
                               0,
                               0,
                               hamParams[1],
                               hamParams[4],
                               0,
                               -hamParams[6]))
    E5,_,_ = run_dmrg(mpo,mbd=mbd,nStates=1,alg='arnoldi')
    E5 = E5/N

    # 2D - x-direction, forwards, staggered 
    Nx = N
    Ny = N
    jr = hamParams[2]*np.ones((Nx,Ny))
    for i in range(Ny):
        jr[i,i-1] = 0
    jl = hamParams[3]*np.ones((Nx,Ny))
    for i in range(Ny):
        jl[i,i-2] = 0
    ju = np.zeros((Nx,Ny))
    jd = np.zeros((Nx,Ny))
    cr = np.zeros((Nx,Ny))
    for i in range(Ny):
        cr[i,i-2] = hamParams[0]
    cl = np.zeros((Nx,Ny))
    for i in range(Ny):
        cl[i,i-1] = hamParams[5]
    cu = np.zeros((Nx,Ny))
    cd = np.zeros((Nx,Ny))
    dr = np.zeros((Nx,Ny))
    for i in range(Ny):
        dr[i,i-1] = hamParams[4]
    dl = np.zeros((Nx,Ny))
    for i in range(Ny):
        dl[i,i-2] = hamParams[1]
    du = np.zeros((Nx,Ny))
    dd = np.zeros((Nx,Ny))
    sy = np.zeros((Nx,Ny))
    sx = hamParams[6]*np.ones((Nx,Ny))
    mpo = return_mpo_asep2D(N,(jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy),periodicx=True)
    E6,_,_ = run_dmrg(mpo,mbd=mbd,nStates=1,alg='arnoldi')
    E6 = E6/N

    # 2D - x-direction, backwards, staggered 
    Nx = N
    Ny = N
    jr = hamParams[3]*np.ones((Nx,Ny))
    for i in range(Ny):
        jr[i,i-1] = 0
    jl = hamParams[2]*np.ones((Nx,Ny))
    for i in range(Ny):
        jl[i,i-2] = 0
    ju = np.zeros((Nx,Ny))
    jd = np.zeros((Nx,Ny))
    cr = np.zeros((Nx,Ny))
    for i in range(Ny):
        cr[i,i-2] = hamParams[5]
    cl = np.zeros((Nx,Ny))
    for i in range(Ny):
        cl[i,i-1] = hamParams[0]
    cu = np.zeros((Nx,Ny))
    cd = np.zeros((Nx,Ny))
    dr = np.zeros((Nx,Ny))
    for i in range(Ny):
        dr[i,i-1] = hamParams[1]
    dl = np.zeros((Nx,Ny))
    for i in range(Ny):
        dl[i,i-2] = hamParams[4]
    du = np.zeros((Nx,Ny))
    dd = np.zeros((Nx,Ny))
    sy = np.zeros((Nx,Ny))
    sx = -hamParams[6]*np.ones((Nx,Ny))
    mpo = return_mpo_asep2D(N,(jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy),periodicx=True)
    E7,_,_ = run_dmrg(mpo,mbd=mbd,nStates=1,alg='arnoldi')
    E7 = E7/N

    # 2D - y-direction, upwards, staggered 
    Nx = N
    Ny = N
    jr = np.zeros((Nx,Ny))
    jl = np.zeros((Nx,Ny))
    ju = hamParams[2]*np.ones((Nx,Ny))
    for i in range(Nx):
        ju[i-1,i] = 0
    jd = hamParams[3]*np.ones((Nx,Ny))
    for i in range(Nx):
        jd[i,i] = 0
    cr = np.zeros((Nx,Ny))
    cl = np.zeros((Nx,Ny))
    cu = np.zeros((Nx,Ny))
    for i in range(Nx):
        cu[i,i] = hamParams[0]
    cd = np.zeros((Nx,Ny))
    for i in range(Nx):
        cd[i-1,i] = hamParams[5]
    dr = np.zeros((Nx,Ny))
    dl = np.zeros((Nx,Ny))
    du = np.zeros((Nx,Ny))
    for i in range(Nx):
        du[i-1,i] = hamParams[4]
    dd = np.zeros((Nx,Ny))
    for i in range(Nx):
        dd[i,i] = hamParams[1]
    sx = np.zeros((Nx,Ny))
    sy = hamParams[6]*np.ones((Nx,Ny))
    mpo = return_mpo_asep2D(N,(jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy),periodicy=True)
    E8,_,_ = run_dmrg(mpo,mbd=mbd,nStates=1,alg='arnoldi')
    E8 = E8/N

    # 2D - y-direction, upwards, staggered 
    Nx = N
    Ny = N
    jr = np.zeros((Nx,Ny))
    jl = np.zeros((Nx,Ny))
    ju = hamParams[3]*np.ones((Nx,Ny))
    for i in range(Nx):
        ju[i-1,i] = 0
    jd = hamParams[2]*np.ones((Nx,Ny))
    for i in range(Nx):
        jd[i,i] = 0
    cr = np.zeros((Nx,Ny))
    cl = np.zeros((Nx,Ny))
    cu = np.zeros((Nx,Ny))
    for i in range(Nx):
        cu[i,i] = hamParams[5]
    cd = np.zeros((Nx,Ny))
    for i in range(Nx):
        cd[i-1,i] = hamParams[0]
    dr = np.zeros((Nx,Ny))
    dl = np.zeros((Nx,Ny))
    du = np.zeros((Nx,Ny))
    for i in range(Nx):
        du[i-1,i] = hamParams[1]
    dd = np.zeros((Nx,Ny))
    for i in range(Nx):
        dd[i,i] = hamParams[4]
    sx = np.zeros((Nx,Ny))
    sy = -hamParams[6]*np.ones((Nx,Ny))
    mpo = return_mpo_asep2D(N,(jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy),periodicy=True)
    E9,_,_ = run_dmrg(mpo,mbd=mbd,nStates=1,alg='arnoldi')
    E9 = E9/N
    
    return E1,E2,E3,E4,E5,E6,E7,E8,E9
