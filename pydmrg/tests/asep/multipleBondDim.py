from dmrg import *
from mpo.asep import return_mpo as return_mpo_asep
from mpo.asep2D import return_mpo as return_mpo_asep2D
import time

# Run a check to ensure PBC are working by setting up staggered
# SEPs on a 2D lattice in both directions

def run_test():
    N = 10
    mbd = np.array([2,4,6])
    hamParams = (np.random.rand(), 
                 np.random.rand(),
                 np.random.rand(),
                 np.random.rand(),
                 np.random.rand(),
                 np.random.rand(),
                 np.random.rand())


    # 1D For comparison
    mpo = return_mpo_asep(N,hamParams)
    E_exact1,_,_ = run_dmrg(mpo,
                     mbd=mbd,
                     alg='exact',
                     fname='saved_states/tests_exact',
                     nStates=2)
    E_exact2,_,_ = run_dmrg(mpo,
                     mbd=mbd,
                     alg='exact',
                     fname='saved_states/tests_exact',
                     initGuess='saved_states/tests_exact',
                     nStates=2)
    E_davidson1,_,_ = run_dmrg(mpo,
                     mbd=mbd,
                     alg='davidson',
                     fname = 'saved_states/tests_davidson',
                     nStates=2)
    E_davidson2,_,_ = run_dmrg(mpo,
                     mbd=mbd,
                     alg='davidson',
                     fname = 'saved_states/tests_davidson',
                     initGuess = 'saved_states/tests_davidson',
                     nStates=2)
    E_arnoldi1,_,_ = run_dmrg(mpo,
                     mbd=mbd,
                     alg='arnoldi',
                     fname = 'saved_states/tests_arnoldi',
                     nStates=2)
    E_arnoldi2,_,_ = run_dmrg(mpo,
                     mbd=mbd,
                     alg='arnoldi',
                     fname = 'saved_states/tests_arnoldi',
                     initGuess = 'saved_states/tests_arnoldi',
                     nStates=2)
    return E_exact1,E_exact2,E_arnoldi1,E_arnoldi2,E_davidson1,E_davidson2
