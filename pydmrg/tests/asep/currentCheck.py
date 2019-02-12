from dmrg import *
from mpo.asep import return_mpo as return_mpo_asep
from mpo.asep import curr_mpo as return_curr_mpo_asep
from tools.contract import full_contract as contract
import time

def run_test():
    N = 10
    mbd = 10
    ds = 0.0001
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
    curr_mpo = return_curr_mpo_asep(N,hamParams,singleBond=True)
    opCurrS = contract(mpo = curr_mpo,
                      mps = 'saved_states/tests/current_mbd0',
                      lmps='saved_states/tests/current_mbd0_left')
    normCurr = contract(mps = 'saved_states/tests/current_mbd0',
                        lmps='saved_states/tests/current_mbd0_left')
    opCurrS /= normCurr
    opCurrS *= (N+1)
    # Calc Current with operator
    curr_mpo = return_curr_mpo_asep(N,hamParams,singleBond=False)
    opCurr = contract(mpo = curr_mpo,
                      mps = 'saved_states/tests/current_mbd0',
                      lmps='saved_states/tests/current_mbd0_left')
    normCurr = contract(mps = 'saved_states/tests/current_mbd0',
                        lmps='saved_states/tests/current_mbd0_left')
    opCurr /= normCurr
    # Calc Current From derivative
    hamParams[-1] -= ds
    mpo = return_mpo_asep(N,hamParams)
    E1,_,_ = run_dmrg(mpo,
                     mbd=mbd,
                     alg='davidson',
                     nStates=1,
                     fname='saved_states/tests/current',
                     calcLeftState=False)
    hamParams[-1] += 2.*ds
    mpo = return_mpo_asep(N,hamParams)
    E2,_,_ = run_dmrg(mpo,
                     mbd=mbd,
                     alg='davidson',
                     nStates=1,
                     fname='saved_states/tests/current',
                     calcLeftState=False)
    derCurr = (E2-E1)/(2.*ds)
    print('Operator Based Current = {}'.format(opCurr))
    print('Operator Based Current = {}'.format(opCurrS))
    print('Derivative Based Current = {}'.format(derCurr))
    return opCurr,opCurrS,derCurr
