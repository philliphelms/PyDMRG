import numpy as np

######## Inputs ##############################
# Model
alpha = 0.35  # In at left
beta = 2/3    # Exit at right
s = -1        # Exponential weighting
gamma = 0     # Exit at left
delta = 0     # In at right
p = 1         # Jump right
q = 0         # Jump Left
# Optimization
tol = 1e-10
maxIter = 10
##############################################

######## Prereqs #############################
# Create MPS
M = []
M.insert(len(M),np.ones((2,1,2)))
M.insert(len(M),np.ones((2,2,1)))
# Create MPO
Sp = np.array([[0,1],[0,0]])
Sm = np.array([[0,0],[1,0]])
n = np.array([[0,0],[0,1]])
v = np.array([[1,0],[0,0]])
I = np.array([[1,0],[0,1]])
z = np.array([[0,0],[0,0]])
W = []
W.insert(len(W),np.array([[alpha*(np.exp(-s)*Sm-v),np.exp(-s)*Sp,-n,I]]))
#W.insert(len(W),np.array([[I,z,z,z],[Sm,z,z,z],[v,z,z,z],[z,np.exp(-s)*Sp,n,I])) # Check if exponential are correctly distributed
W.insert(len(W),np.array([[I],[Sm],[v],[beta*(np.exp(-s)*Sp-n)]]))
##############################################

# Make MPS Right Canonical ###################
M_reshape = np.swapaxes(M[1],0,1)
M_reshape = np.reshape(M_reshape,(2,2))
(U,s,V) = np.linalg.svd(M_reshape,full_matrices=True)
M_reshape = np.reshape(V,(2,2,1))
M[1] = np.swapaxes(M_reshape,0,1)
M[0] = np.einsum('klj,ji,i->kli',M[0],U,s)
##############################################

# Optimization Sweeps ########################
converged = False
iterCnt = 0
#E = np.einsum('ijk,lmin,npq,rks,mtru,uqv->jlpstv',np.conj(M[0]),W[0],M[0],np.conj(M[1]),W[1],M[1])
E_prev= np.einsum('ijk,lmin,npq,rks,mtru,uqv->',np.conj(M[0]),W[0],M[0],np.conj(M[1]),W[1],M[1])
print('Initial Energy = {}'.format(E_prev))
while not converged:
# Right Sweep ----------------------------
    # Optimization
    #H = np.einsum('jlp,lmin,rks,mtru,uqv->ijpnkq',np.array([[[1]]]),W[0],np.conj(M[1]),W[1],M[1])
    print('\tSite 1')
    H = np.einsum('ijk,jlmn,olp->mionkp',np.array([[[1]]]),W[0],\
            np.einsum('ijk,lmio,opq,kmq->jlp',np.conj(M[1]),W[1],M[1],np.array([[[1]]])))
    H = np.reshape(H,(4,4))
    u,v = np.linalg.eig(H)
    # select max eigenvalue
    max_ind = np.argsort(u)[-1]
    E = u[max_ind]
    v = v[:,max_ind]
    print('\t\tCurrent Energy = {}'.format(E))
    M[0] = np.reshape(v,(2,1,2)) # Could this be wrong?!?!
    # Left Normalize
    print('\t\tCheck Energy = {}'.format(np.einsum('ijk,lmin,npq,rks,mtru,uqv->',np.conj(M[0]),W[0],M[0],np.conj(M[1]),W[1],M[1])))
    M_reshape = np.reshape(M[0],(2,2))
    (U,s,V) = np.linalg.svd(M_reshape,full_matrices=True)
    M[0] = np.reshape(U,(2,1,2))
    M[1] = np.einsum('i,ij,kjl->kil',s,V,M[1])
    print('\t\tCheck Energy = {}'.format(np.einsum('ijk,lmin,npq,rks,mtru,uqv->',np.conj(M[0]),W[0],M[0],np.conj(M[1]),W[1],M[1])))
# Left Sweep -----------------------------
    # Optimization
    print('\tSite 0')
    H = np.einsum('ijk,lmin,npq,mtru,stv->rkqusv',np.conj(M[0]),W[0],M[0],W[1],np.array([[[1]]]))
    H = np.einsum('ijk,jlmn,olp->mionkp',\
            np.einsum('ijk,lmio,opq->kmq',np.conj(M[0]),W[0],M[0]),\
            W[1],np.array([[[1]]]))
    H = np.reshape(H,(4,4))
    u,v = np.linalg.eig(H)
    # select max eigenvalue
    max_ind = np.argsort(u)[-1]
    E = u[max_ind]
    v = v[:,max_ind]
    print('\t\tCurrent Energy = {}'.format(E))
    M[1] = np.reshape(v,(2,2,1)) # Could this be wrong?!?!
    # Right Normalize
    print('\t\tCheck Energy = {}'.format(np.einsum('ijk,lmin,npq,rks,mtru,uqv->',np.conj(M[0]),W[0],M[0],np.conj(M[1]),W[1],M[1])))
    M_reshape = np.swapaxes(M[1],0,1)
    M_reshape = np.reshape(M_reshape,(2,2))
    (U,s,V) = np.linalg.svd(M_reshape,full_matrices=True)
    M_reshape = np.reshape(V,(2,2,1))
    M[1] = np.swapaxes(M_reshape,0,1)
    M[0] = np.einsum('klj,ji,i->kli',M[0],U,s)
    print('\t\tCheck Energy = {}'.format(np.einsum('ijk,lmin,npq,rks,mtru,uqv->',np.conj(M[0]),W[0],M[0],np.conj(M[1]),W[1],M[1])))
# Convergence Test -----------------------
    if np.abs(E-E_prev) < tol:
        print('#'*75+'\nConverged at E = {}'.format(E)+'\n'+'#'*75)
        converged = True
    elif iterCnt > maxIter:
        print('Convergence not acheived')
        converged = True
    else:
        iterCnt += 1
        E_prev = np.einsum('ijk,lmin,npq,rks,mtru,uqv->',np.conj(M[0]),W[0],M[0],np.conj(M[1]),W[1],M[1])
##############################################
