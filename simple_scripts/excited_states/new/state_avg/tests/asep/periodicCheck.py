from dmrg import *
from mpo.asep2D import return_mpo
import time

# Run a check to ensure PBC are working by setting up staggered
# SEPs on a 2D lattice in both directions

# 1D For comparison
N = 4
hamParams = (np.random.rand, 
             np.random.rand,
             np.random.rand,
             np.random.rand,
             np.random.rand,
             np.random.rand,
             np.random.rand)


