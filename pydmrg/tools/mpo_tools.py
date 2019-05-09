import numpy as np
import copy

def mpo2mat(mpo):
    N = len(mpo[0])
    mat = np.zeros((2**N,2**N))
    for i in range(2**N):
        i_occ = list(map(lambda x: int(x),'0'*(N-len(bin(i)[2:]))+bin(i)[2:]))
        for j in range(2**N):
            j_occ = list(map(lambda x: int(x),'0'*(N-len(bin(j)[2:]))+bin(j)[2:]))
            for k in range(len(mpo)):
                tmp_mat = np.array([[1.]])
                for l in range(N):
                    if mpo[k][l] is not None:
                        tmp_mat = np.einsum('ij,jk->ik',tmp_mat,mpo[k][l][:,:,i_occ[l],j_occ[l]])
                    else:
                        multiplier = np.array([[np.eye(2)]])
                        tmp_mat = np.einsum('ij,jk->ik',tmp_mat,multiplier[:,:,i_occ[l],j_occ[l]])
                mat[i,j] += tmp_mat[[0]]
    return mat

def mpo_conj_trans(mpo):
    # Return the conjugate transpose of the mp
    mpoct = copy.deepcopy(mpo)
    for opind,op in enumerate(mpo):
        for site in range(len(op)):
            if mpo[opind][site] is None:
                pass
            else:
                mpoct[opind][site] = np.transpose(mpo[opind][site],(0,1,3,2)).conj()
    return mpoct
