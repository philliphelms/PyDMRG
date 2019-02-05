from dmrg import *
from mpo.asep import return_mpo as return_mpo_asep
from mpo.asep import curr_mpo as return_curr_mpo_asep
from mpo.asep2D import return_mpo as return_mpo_asep2D
from mpo.asep2D import curr_mpo as return_curr_mpo_asep2D
from tools.contract import full_contract as contract
import time

# Run a check to ensure PBC are working by setting up staggered
# SEPs on a 2D lattice in both directions
def run_test():
    N = 3
    mbd = 10
    ##########################################################################
    # 1D Current Baseline
    hamParams0 = np.array([np.random.rand(), # in on left
                 np.random.rand(), # Out on left
                 np.random.rand(), # forward hop
                 np.random.rand(), # backward hop
                 np.random.rand(), # Out on right
                 np.random.rand(), # In on right
                 np.random.rand()]) # Bias
    mpo = return_mpo_asep(N,hamParams0)
    E1,_,_ = run_dmrg(mpo,
                      mbd=mbd,
                      nStates=1,
                      fname='saved_states/tests/current2D',
                      calcLeftState=True)
    # Evaluate Current via operator
    curr_mpo = return_curr_mpo_asep(N,hamParams0)
    currContract1 = contract(mpo = curr_mpo,
                            mps = 'saved_states/tests/current2D_mbd0',
                            lmps='saved_states/tests/current2D_mbd0_left')
    normCurr1 = contract(mps = 'saved_states/tests/current2D_mbd0',
                        lmps='saved_states/tests/current2D_mbd0_left')
    currContract1 /= normCurr1
    
    ##########################################################################
    # 1D Derivative current
    ds = 0.001
    hamParams0[-1] += ds
    mpo = return_mpo_asep(N,hamParams0)
    E1d,_,_ = run_dmrg(mpo,
                      mbd=mbd,
                      nStates=1,
                      fname='saved_states/tests/current2D',
                      calcLeftState=True)
    hamParams0[-1] -= ds
    ##########################################################################
    # 2D For (Vertically Stacked)
    Nx,Ny=N,N
    hamParams = np.array([hamParams0[2],
                          hamParams0[3],
                          0.,
                          0.,
                          hamParams0[0],
                          hamParams0[5],
                          0.,
                          0.,
                          hamParams0[4],
                          hamParams0[1],
                          0.,
                          0.,
                          hamParams0[6],0.])
    mpo = return_mpo_asep2D([Nx,Ny],hamParams)
    E2,_,_ = run_dmrg(mpo,
                      mbd=mbd,
                      nStates=1,
                      fname='saved_states/tests/current2D',
                      calcLeftState=True)
    # Evaluate Current via operator
    curr_mpo = return_curr_mpo_asep2D([Nx,Ny],hamParams)
    currContract2 = contract(mpo = curr_mpo,
                            mps = 'saved_states/tests/current2D_mbd0',
                            lmps= 'saved_states/tests/current2D_mbd0_left')
    normCurr2 = contract(mps = 'saved_states/tests/current2D_mbd0',
                        lmps= 'saved_states/tests/current2D_mbd0_left')
    currContract2 /= normCurr2
    ##########################################################################
    # 2D Vertically Stacked backwards
    hamParams = np.array([hamParams0[3],
                          hamParams0[2],
                          0.,
                          0.,
                          hamParams0[5],
                          hamParams0[0],
                          0.,
                          0.,
                          hamParams0[1],
                          hamParams0[4],
                          0.,
                          0.,
                          -hamParams0[6],
                          0.])
    mpo = return_mpo_asep2D([Nx,Ny],hamParams)
    E3,_,_ = run_dmrg(mpo,
                      mbd=mbd,
                      nStates=1,
                      fname='saved_states/tests/current2D',
                      calcLeftState=True)
    # Evaluate Current via operator
    curr_mpo = return_curr_mpo_asep2D([Nx,Ny],hamParams)
    currContract3 = contract(mpo = curr_mpo,
                            mps = 'saved_states/tests/current2D_mbd0',
                            lmps= 'saved_states/tests/current2D_mbd0_left')
    normCurr3 = contract(mps = 'saved_states/tests/current2D_mbd0',
                        lmps= 'saved_states/tests/current2D_mbd0_left')
    currContract3 /= normCurr3
    ##########################################################################
    # 2D Horizontally Stacked 
    hamParams = np.array([0.,
                          0.,
                          hamParams0[3],
                          hamParams0[2],
                          0.,
                          0.,
                          hamParams0[5],
                          hamParams0[0],
                          0.,
                          0.,
                          hamParams0[1],
                          hamParams0[4],
                          0.,
                          -hamParams0[6]])
    mpo = return_mpo_asep2D([Nx,Ny],hamParams)
    E4,_,_ = run_dmrg(mpo,
                      mbd=mbd,
                      nStates=1,
                      fname='saved_states/tests/current2D',
                      calcLeftState=True)
    # Evaluate Current via operator
    curr_mpo = return_curr_mpo_asep2D([Nx,Ny],hamParams)
    currContract4 = contract(mpo = curr_mpo,
                            mps = 'saved_states/tests/current2D_mbd0',
                            lmps= 'saved_states/tests/current2D_mbd0_left')
    normCurr4 = contract(mps = 'saved_states/tests/current2D_mbd0',
                        lmps= 'saved_states/tests/current2D_mbd0_left')
    currContract4 /= normCurr4
    ##########################################################################
    # 2D Horizontally Stacked Backwards 
    hamParams = np.array([0.,
                          0.,
                          hamParams0[2],
                          hamParams0[3],
                          0.,
                          0.,
                          hamParams0[0],
                          hamParams0[5],
                          0.,
                          0.,
                          hamParams0[4],
                          hamParams0[1],
                          0.,
                          hamParams0[6]])
    mpo = return_mpo_asep2D([Nx,Ny],hamParams)
    E5,_,_ = run_dmrg(mpo,
                      mbd=mbd,
                      nStates=1,
                      fname='saved_states/tests/current2D',
                      calcLeftState=True)
    # Evaluate Current via operator
    curr_mpo = return_curr_mpo_asep2D([Nx,Ny],hamParams)
    currContract5 = contract(mpo = curr_mpo,
                            mps = 'saved_states/tests/current2D_mbd0',
                            lmps= 'saved_states/tests/current2D_mbd0_left')
    normCurr5 = contract(mps = 'saved_states/tests/current2D_mbd0',
                        lmps= 'saved_states/tests/current2D_mbd0_left')
    currContract5 /= normCurr5
    ##########################################################################
    # 2D Vertical Staggered
    jr = hamParams0[2]*np.ones((Nx,Ny))
    for i in range(Ny):
        jr[i,i-1] = 0
    jl = hamParams0[3]*np.ones((Nx,Ny))
    for i in range(Ny):
        jl[i,i-2] = 0
    ju = np.zeros((Nx,Ny))
    jd = np.zeros((Nx,Ny))
    cr = np.zeros((Nx,Ny))
    for i in range(Ny):
        cr[i,i-2] = hamParams0[0]
    cl = np.zeros((Nx,Ny))
    for i in range(Ny):
        cl[i,i-1] = hamParams0[5]
    cu = np.zeros((Nx,Ny))
    cd = np.zeros((Nx,Ny))
    dr = np.zeros((Nx,Ny))
    for i in range(Ny):
        dr[i,i-1] = hamParams0[4]
    dl = np.zeros((Nx,Ny))
    for i in range(Ny):
        dl[i,i-2] = hamParams0[1]
    du = np.zeros((Nx,Ny))
    dd = np.zeros((Nx,Ny))
    sy = np.zeros((Nx,Ny))
    sx = hamParams0[6]*np.ones((Nx,Ny))
    hamParams = (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy)
    mpo = return_mpo_asep2D([Nx,Ny],hamParams,periodicx=True)
    E6,_,_ = run_dmrg(mpo,
                      mbd=mbd,
                      nStates=1,
                      fname='saved_states/tests/current2D',
                      calcLeftState=True)
    # Evaluate Current via operator
    curr_mpo = return_curr_mpo_asep2D([Nx,Ny],hamParams,periodicx=True)
    currContract6 = contract(mpo = curr_mpo,
                            mps = 'saved_states/tests/current2D_mbd0',
                            lmps= 'saved_states/tests/current2D_mbd0_left')
    normCurr6 = contract(mps = 'saved_states/tests/current2D_mbd0',
                        lmps= 'saved_states/tests/current2D_mbd0_left')
    currContract6 /= normCurr6
    ##########################################################################
    # 2D Vertical Staggered
    jr = np.zeros((Nx,Ny))
    jl = np.zeros((Nx,Ny))
    ju = hamParams0[2]*np.ones((Nx,Ny))
    for i in range(Nx):
        ju[i-1,i] = 0
    jd = hamParams0[3]*np.ones((Nx,Ny))
    for i in range(Nx):
        jd[i,i] = 0
    cr = np.zeros((Nx,Ny))
    cl = np.zeros((Nx,Ny))
    cu = np.zeros((Nx,Ny))
    for i in range(Nx):
        cu[i,i] = hamParams0[0]
    cd = np.zeros((Nx,Ny))
    for i in range(Nx):
        cd[i-1,i] = hamParams0[5]
    dr = np.zeros((Nx,Ny))
    dl = np.zeros((Nx,Ny))
    du = np.zeros((Nx,Ny))
    for i in range(Nx):
        du[i-1,i] = hamParams0[4]
    dd = np.zeros((Nx,Ny))
    for i in range(Nx):
        dd[i,i] = hamParams0[1]
    sx = np.zeros((Nx,Ny))
    sy = hamParams0[6]*np.ones((Nx,Ny))
    hamParams = (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy)
    mpo = return_mpo_asep2D([Nx,Ny],hamParams,periodicy=True)
    E6,_,_ = run_dmrg(mpo,
                      mbd=mbd,
                      nStates=1,
                      fname='saved_states/tests/current2D',
                      calcLeftState=True)
    # Evaluate Current via operator
    curr_mpo = return_curr_mpo_asep2D([Nx,Ny],hamParams,periodicy=True)
    currContract7 = contract(mpo = curr_mpo,
                            mps = 'saved_states/tests/current2D_mbd0',
                            lmps= 'saved_states/tests/current2D_mbd0_left')
    normCurr7 = contract(mps = 'saved_states/tests/current2D_mbd0',
                        lmps= 'saved_states/tests/current2D_mbd0_left')
    currContract7 /= normCurr7
    ##########################################################################
    # Print Results
    print('1D Current = {}'.format(currContract1))
    print('1D Current Derivative = {}'.format((E1-E1d)/(ds)))
    print('2D Current (per lane) = {}/{} = {}'.format(currContract2,Nx,currContract2/Nx))
    print('2D Current (per lane) backwards= {}/{} = {}'.format(currContract3,Nx,currContract3/Nx))
    print('2D Current (per lane) flipped backwards= {}/{} = {}'.format(currContract4,Nx,currContract4/Nx))
    print('2D Current (per lane) flipped= {}/{} = {}'.format(currContract5,Nx,currContract5/Nx))
    print('2D Current Staggered (per lane) = {}/{} = {}'.format(currContract6,Nx,currContract6/Nx))
    print('2D Current Staggered (per lane) flipped = {}/{} = {}'.format(currContract7,Nx,currContract7/Nx))
    """
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
    """

if __name__ == "__main__":
    run_test()
