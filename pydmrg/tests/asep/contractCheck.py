from dmrg import *
from mpo.asep import return_mpo as return_mpo_asep
from mpo.asep import curr_mpo as return_curr_mpo_asep
from tools.contract import full_contract as contract
import time

def run_test():
    N = 10
    mbd = 10
    ds = 0.01
    hamParams = np.array([np.random.rand(), 
                         np.random.rand(),
                         np.random.rand(),
                         np.random.rand(),
                         np.random.rand(),
                         np.random.rand(),
                         np.random.rand()])


    # 1D For comparison
    mpo = return_mpo_asep(N,hamParams)
    E,_,_ = run_dmrg(mpo,
                     mbd=mbd,
                     alg='davidson',
                     nStates=1,
                     fname='saved_states/tests/current',
                     calcLeftState=True)
    
    # Calc Current with operator
    Econt1 = contract(mpo = mpo,
                      mps = 'saved_states/tests/current_mbd0')
    norm1 = contract(mps = 'saved_states/tests/current_mbd0')
    Econt2 = contract(mpo = mpo,
                      lmps = 'saved_states/tests/current_mbd0_left')
    norm2 = contract(lmps = 'saved_states/tests/current_mbd0_left')
    Econt3 = contract(mpo = mpo,
                      mps = 'saved_states/tests/current_mbd0',
                      lmps = 'saved_states/tests/current_mbd0_left')
    norm3 = contract(mps = 'saved_states/tests/current_mbd0',
                     lmps = 'saved_states/tests/current_mbd0_left')
    print('Actual Energy = {}'.format(E))
    print('Contract Right Energy = {}'.format(Econt1/norm1))
    print('Contract Left Energy = {}'.format(Econt2/norm2))
    print('Contract Left Right Energy = {}'.format(Econt3/norm3))
    return E,Econt1/norm1,Econt2/norm2,Econt3/norm3
