import numpy as np
from mpo.ops import *
import collections

############################################################################
# Ising Model
#
# A very simple implementation of the ising model mpo
###########################################################################

def return_mpo(N,hamParams):
    h = hamParams[0]
    print('h = {}'.format(h))
    # List to hold all mpos
    mpoL = []
    # Mainmpo
    mpo = [None]*N
    for site in range(N):
        gen_mpo = np.array([[I,   z, z],
                            [X,   z, z],
                            [h*Z, X, I]])
        if (site == 0):
            mpo[site] = np.expand_dims(gen_mpo[-1,:],0)
        elif (site == N-1):
            mpo[site] = np.expand_dims(gen_mpo[:,0],1)
        else:
            mpo[site] = gen_mpo
    mpoL.append(mpo)
    return mpoL
