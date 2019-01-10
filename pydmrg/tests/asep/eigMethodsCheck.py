from dmrg import *
from mpo.asep import return_mpo as return_mpo_asep
from mpo.asep2D import return_mpo as return_mpo_asep2D
import time

# Run a check to ensure PBC are working by setting up staggered
# SEPs on a 2D lattice in both directions

def run_test():
    N = 10
    mbd = 10
    hamParams = (np.random.rand(), 
                 np.random.rand(),
                 np.random.rand(),
                 np.random.rand(),
                 np.random.rand(),
                 np.random.rand(),
                 np.random.rand())


    # 1D For comparison
    mpo = return_mpo_asep(N,hamParams)
    E_exact,_,_ = run_dmrg(mpo,
                     mbd=mbd,
                     alg='exact',
                     nStates=1)
    E_arnoldi,_,_ = run_dmrg(mpo,
                     mbd=mbd,
                     alg='arnoldi',
                     nStates=1)
    E_davidson,_,_ = run_dmrg(mpo,
                     mbd=mbd,
                     alg='davidson',
                     nStates=1)
    return E_exact,E_arnoldi,E_davidson
