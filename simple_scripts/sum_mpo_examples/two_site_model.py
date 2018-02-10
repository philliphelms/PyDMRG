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
# Create all generic operations
Sp = np.array([[0,1],[0,0]])
Sm = np.array([[0,0],[1,0]])
n = np.array([[0,0],[0,1]])
v = np.array([[1,0],[0,0]])
I = np.array([[1,0],[0,1]])
z = np.array([[0,0],[0,0]])
# Container to hold all operators
ops = []
# First Operator ----------------------------
op1 = []
# Site 1
op1.insert(len(op1),alpha*(np.exp(-s)*Sm-v))
# Site 2
op1.insert(len(op1),I)
# Add to operator list
ops.insert(len(ops),op1)
# Second Operator ---------------------------
op2 = []
# Site 1
op2.insert(len(op2),np.exp(-s)*Sp)
# Site 2
op2.insert(len(op2),Sm)
# Add to operator list
ops.insert(len(ops),op2)
# Third Operator ---------------------------
op3 = []
# Site 1
op3.insert(len(op3),-n)
# Site 2
op3.insert(len(op3),v)
# Add to operator list
ops.insert(len(ops),op3)
# Fourth Operator ---------------------------
op4 = []
# Site 1
op4.insert(len(op4),I)
# Site 2
op4.insert(len(op4),beta*(np.exp(-s)*Sp-n))
# Add to operator list
ops.insert(len(ops),op4)

nops = len(ops)
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
E_prev = 0
for i in range(nops):
    E_prev += np.einsum('ijk,in,npq,rks,ru,uqv->',np.conj(M[0]),ops[i][0],M[0],np.conj(M[1]),ops[i][1],M[1])
print('Initial Energy = {}'.format(E_prev))
while not converged:
# Right Sweep ----------------------------
    # Optimization
    print('\tSite 0')
    #H = np.einsum('ijk,in,npq,ru,stv->rkqusv',np.conj(M[0]),ops[i][0],M[0],ops[i][1],np.array([[[1]]]))
    for i in range(nops):
        if i == 0:
            tmp_arr = np.einsum('ijk,io,opq->kq',np.conj(M[0]),ops[i][0],M[0])
            H = np.einsum('ik,mn,op->mionkp',tmp_arr,ops[i][1],np.array([[1]]))
        else:
            tmp_arr = np.einsum('ijk,io,opq->kq',np.conj(M[0]),ops[i][0],M[0])
            H += np.einsum('ik,mn,op->mionkp',tmp_arr,ops[i][1],np.array([[1]]))
    H = np.reshape(H,(4,4))
    u,v = np.linalg.eig(H)
    # select max eigenvalue
    max_ind = np.argsort(u)[-1]
    E = u[max_ind]
    v = v[:,max_ind]
    print('\t\tCurrent Energy = {}'.format(E))
    M[1] = np.reshape(v,(2,2,1)) # Could this be wrong?!?!
    # Right Normalize
    M_reshape = np.swapaxes(M[1],0,1)
    M_reshape = np.reshape(M_reshape,(2,2))
    (U,s,V) = np.linalg.svd(M_reshape,full_matrices=True)
    M_reshape = np.reshape(V,(2,2,1))
    M[1] = np.swapaxes(M_reshape,0,1)
    M[0] = np.einsum('klj,ji,i->kli',M[0],U,s)
# Left Sweep -----------------------------
    # Optimization
    print('\tSite 1')
    for i in range(nops):
        if i == 0:
            tmp_arr = np.einsum('ijk,io,opq,kq->jp',np.conj(M[1]),ops[i][1],M[1],np.array([[1]]))
            H = np.einsum('ik,mn,op->mionkp',np.array([[1]]),ops[i][0],tmp_arr)
        else:
            tmp_arr = np.einsum('ijk,io,opq,kq->jp',np.conj(M[1]),ops[i][1],M[1],np.array([[1]]))
            H += np.einsum('ik,mn,op->mionkp',np.array([[1]]),ops[i][0],tmp_arr)
    H = np.reshape(H,(4,4))
    u,v = np.linalg.eig(H)
    # select max eigenvalue
    max_ind = np.argsort(u)[-1]
    E = u[max_ind]
    v = v[:,max_ind]
    print('\t\tCurrent Energy = {}'.format(E))
    M[0] = np.reshape(v,(2,1,2)) # Could this be wrong?!?!
    # Left Normalize
    M_reshape = np.reshape(M[0],(2,2))
    (U,s,V) = np.linalg.svd(M_reshape,full_matrices=True)
    M[0] = np.reshape(U,(2,1,2))
    M[1] = np.einsum('i,ij,kjl->kil',s,V,M[1])
# Convergence Test -----------------------
    if np.abs(E-E_prev) < tol:
        print('#'*75+'\nConverged at E = {}'.format(E)+'\n'+'#'*75)
        converged = True
    elif iterCnt > maxIter:
        print('Convergence not acheived')
        converged = True
    else:
        iterCnt += 1
        E_prev = E
##############################################
