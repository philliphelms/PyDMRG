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
    hamParams = (np.random.rand(), # in on left
                 np.random.rand(), # Out on left
                 np.random.rand(), # forward hop
                 np.random.rand(), # backward hop
                 np.random.rand(), # Out on right
                 np.random.rand(), # In on right
                 np.random.rand()) # Bias
    # 1D For comparison ---------------------------------------------
    mpo = return_mpo_asep(N,hamParams)
    E1,_,_ = run_dmrg(mpo,
                      mbd=mbd,
                      fname='saved_states/tests/current2D',
                      nStates=1,
                      calcLeftState=True)
    curr_mpo = return_curr_mpo_asep(N,hamParams)
    currContract = contract(N,mbd,
                            mpo = curr_mpo,
                            mps = 'saved_states/tests/current_mbd0',
                            lmps='saved_states/tests/current_mbd0_left')
    normCurr = contract(N,mbd,
                        mps = 'saved_states/tests/current_mbd0',
                        lmps='saved_states/tests/current_mbd0_left')
    currContract /= normCurr
    # 2D Current ------------------------------------------------------
    Nx = N
    Ny = N
    hamParams2D = np.array([hamParams[2],hamParams[3],0,0,hamParams[0],hamParams[5],0,0,hamParams[4],hamParams[1],0,0,hamParams[6],0])
    mpo = return_mpo_asep2D(N,hamParams2D)
    E2,_,_ = run_dmrg(mpo,
                      mbd=mbd,
                      nStates=1,
                      fname = 'saved_states/tests/current2D',
                      alg='davidson',
                      calcLeftState=True)
    curr_mpo = return_curr_mpo_asep2D(N,hamParams2D)
    currContract2D = contract(N,mbd,
                            mpo = curr_mpo,
                            mps = 'saved_states/tests/current2D_mbd0',
                            lmps='saved_states/tests/current2D_mbd0_left')
    normCurr = contract(N,mbd,
                        mps = 'saved_states/tests/current2D_mbd0',
                        lmps='saved_states/tests/current2D_mbd0_left')
    currContract2D /= normCurr
    print('1D Current = {}, 2D Current = {}'.format(currContract,currContract2D/N))
    return currContract,currContract2D/N
