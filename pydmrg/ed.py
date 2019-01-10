import numpy as np
from tools.mpo_tools import mpo2mat
from scipy.linalg import eig

def ed(mpo,left=False):
    # Convert mpo to matrix
    M = mpo2mat(mpo)
    # Solve Eigenproblem
    if left:
        e,vl,vr = eig(M,left=True)
        inds = np.argsort(np.real(e))[::-1]
        return e[inds],vl[:,inds],vr[:,inds]
    else:
        e,v = eig(M)
        inds = np.argsort(np.real(e))[::-1]
        return e[inds],v[:,inds]
