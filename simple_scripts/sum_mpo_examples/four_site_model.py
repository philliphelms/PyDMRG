import numpy as np

######## Inputs ##############################
# Model
N = 4
alpha = 0.35  # In at left
beta = 2/3    # Exit at right
s = -1        # Exponential weighting
gamma = 0     # Exit at left
delta = 0     # In at right
p = 1         # Jump right
q = 0         # Jump Left
# Optimization
tol = 1e-5
maxIter = 10
##############################################

######## Prereqs #############################
# Create MPS
M = []
M.insert(len(M),np.ones((2,1,2)))
M.insert(len(M),np.ones((2,2,4)))
M.insert(len(M),np.ones((2,4,2)))
M.insert(len(M),np.ones((2,2,1)))
# Create MPO
Sp = np.array([[0,1],[0,0]])
Sm = np.array([[0,0],[1,0]])
n = np.array([[0,0],[0,1]])
v = np.array([[1,0],[0,0]])
I = np.array([[1,0],[0,1]])
z = np.array([[0,0],[0,0]])
# container to hold all operators
ops = []
# First Operator ----------------------------
ops.insert(len(ops),[alpha*(np.exp(-s)*Sm-v),I,I,I])
# Middle Operators ----------------------------
ops.insert(len(ops),[np.exp(-s)*Sp,Sm,I,I])
ops.insert(len(ops),[-n,v,I,I])
ops.insert(len(ops),[I,np.exp(-s)*Sp,Sm,I])
ops.insert(len(ops),[I,-n,v,I])
ops.insert(len(ops),[I,I,np.exp(-s)*Sp,Sm])
ops.insert(len(ops),[I,I,-n,v])
# Exit Operator ------------------------------
ops.insert(len(ops),[I,I,I,beta*(np.exp(-s)*Sp-n)])

nops = len(ops)
##############################################
# Create F
F = []
for i in range(nops):
    F_tmp = []
    F_tmp.insert(len(F_tmp),np.array([[1]]))
    F_tmp.insert(len(F_tmp),np.zeros((2,2)))
    F_tmp.insert(len(F_tmp),np.zeros((4,4)))
    F_tmp.insert(len(F_tmp),np.zeros((2,2)))
    F_tmp.insert(len(F_tmp),np.array([[1]]))
    F.insert(len(F),F_tmp)
##############################################

# Make MPS Right Canonical ###################
# Site 3
M_reshape = np.swapaxes(M[3],0,1)
M_reshape = np.reshape(M_reshape,(2,2))
(U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
M_reshape = np.reshape(V,(2,2,1))
M[3] = np.swapaxes(M_reshape,0,1)
M[2] = np.einsum('klj,ji,i->kli',M[2],U,s)
# Site 2
M_reshape = np.swapaxes(M[2],0,1)
M_reshape = np.reshape(M_reshape,(4,4))
(U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
M_reshape = np.reshape(V,(4,2,2))
M[2] = np.swapaxes(M_reshape,0,1)
M[1] = np.einsum('klj,ji,i->kli',M[1],U,s)
# Site 1
M_reshape = np.swapaxes(M[1],0,1)
M_reshape = np.reshape(M_reshape,(2,8))
(U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
M_reshape = np.reshape(V,(2,2,4))
M[1] = np.swapaxes(M_reshape,0,1)
M[0] = np.einsum('klj,ji,i->kli',M[0],U,s)
##############################################

# Calculate Initial F ########################
for i in range(nops):
    F[i][3] = np.einsum('bxc,be,eaf,cf->xa',np.conj(M[3]),ops[i][3],M[3],F[i][4])
    F[i][2] = np.einsum('wsx,wz,zva,xa->sv',np.conj(M[2]),ops[i][2],M[2],F[i][3])
    F[i][1] = np.einsum('rks,ru,uqv,sv->kq',np.conj(M[1]),ops[i][1],M[1],F[i][2])
##############################################

# Optimization Sweeps ########################
converged = False
iterCnt = 0
#E = np.einsum('ijk,lmin,npq,rks,mtru,uqv->jlpstv',np.conj(M[0]),W[0],M[0],np.conj(M[1]),W[1],M[1])
E_prev = 0
for i in range(nops):
    E_prev += np.einsum('jp,ijk,in,npq,rks,ru,uqv,wsx,wz,zva,bxc,be,eaf,cf->',\
                      np.array([[1]]),\
                      np.conj(M[0]),ops[i][0],M[0],\
                      np.conj(M[1]),ops[i][1],M[1],\
                      np.conj(M[2]),ops[i][2],M[2],\
                      np.conj(M[3]),ops[i][3],M[3],\
                      np.array([[1]]))
print('Initial Energy = {}'.format(E_prev))
while not converged:
# Right Sweep ----------------------------
    print('Right Sweep {}'.format(iterCnt))
    # Optimization
    print('\tSite 0')
    for i in range(nops):
        if i == 0:
            H = np.einsum('jp,in,kq->ijknpq',F[i][0],ops[i][0],F[i][1])
        else:
            H += np.einsum('jp,in,kq->ijknpq',F[i][0],ops[i][0],F[i][1])
    H = np.reshape(H,(4,4))
    u,v = np.linalg.eig(H)
    # select max eigenvalue
    max_ind = np.argsort(u)[-1]
    E = u[max_ind]
    v = v[:,max_ind]
    print('\t\tCurrent Energy = {}'.format(E))
    M[0] = np.reshape(v,(2,1,2))
    # Right Normalize
    M_reshape = np.reshape(M[0],(2,2))
    (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
    M[0] = np.reshape(U,(2,1,2))
    M[1] = np.einsum('i,ij,kjl->kil',s,V,M[1])
    # Update F
    for i in range(nops):
        F[i][1] = np.einsum('jp,ijk,in,npq->kq',F[i][0],np.conj(M[0]),ops[i][0],M[0])
    # NEXT SITE
    print('\tSite 1')
    for i in range(nops):
        if i == 0:
            H = np.einsum('jp,in,kq->ijknpq',F[i][1],ops[i][1],F[i][2])
        else:
            H += np.einsum('jp,in,kq->ijknpq',F[i][1],ops[i][1],F[i][2])
    H = np.reshape(H,(16,16))
    u,v = np.linalg.eig(H)
    # select max eigenvalue
    max_ind = np.argsort(u)[-1]
    E = u[max_ind]
    v = v[:,max_ind]
    print('\t\tCurrent Energy = {}'.format(E))
    M[1] = np.reshape(v,(2,2,4))
    # Right Normalize
    M_reshape = np.reshape(M[1],(4,4))
    (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
    M[1] = np.reshape(U,(2,2,4))
    M[2] = np.einsum('i,ij,kjl->kil',s,V,M[2])
    # Update F
    for i in range(nops):
        F[i][2] = np.einsum('jp,ijk,in,npq->kq',F[i][1],np.conj(M[1]),ops[i][1],M[1])
    # NEXT SITE
    print('\tSite 2')
    for i in range(nops):
        if i == 0:
            H = np.einsum('jp,in,kq->ijknpq',F[i][2],ops[i][2],F[i][3])
        else:
            H += np.einsum('jp,in,kq->ijknpq',F[i][2],ops[i][2],F[i][3])
    H = np.reshape(H,(16,16))
    u,v = np.linalg.eig(H)
    # select max eigenvalue
    max_ind = np.argsort(u)[-1]
    E = u[max_ind]
    v = v[:,max_ind]
    print('\t\tCurrent Energy = {}'.format(E))
    M[2] = np.reshape(v,(2,4,2))
    # Right Normalize
    M_reshape = np.reshape(M[2],(8,2))
    (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
    M[2] = np.reshape(U,(2,4,2))
    M[3] = np.einsum('i,ij,kjl->kil',s,V,M[3])
    # Update F
    for i in range(nops):
        F[i][3] = np.einsum('jp,ijk,in,npq->kq',F[i][2],np.conj(M[2]),ops[i][2],M[2])
# Left Sweep -----------------------------
    print('Left Sweep {}'.format(iterCnt))
    # Optimization
    print('\tSite 3')
    for i in range(nops):
        if i == 0:
            H = np.einsum('jp,in,kq->ijknpq',F[i][3],ops[i][3],F[i][4])
        else:
            H += np.einsum('jp,in,kq->ijknpq',F[i][3],ops[i][3],F[i][4])
    H = np.reshape(H,(4,4))
    u,v = np.linalg.eig(H)
    # select max eigenvalue
    max_ind = np.argsort(u)[-1]
    E = u[max_ind]
    v = v[:,max_ind]
    print('\t\tCurrent Energy = {}'.format(E))
    M[3] = np.reshape(v,(2,2,1))
    # Right Normalize 
    M_reshape = np.swapaxes(M[3],0,1)
    M_reshape = np.reshape(M_reshape,(2,2))
    (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
    M_reshape = np.reshape(V,(2,2,1))
    M[3] = np.swapaxes(M_reshape,0,1)
    M[2] = np.einsum('klj,ji,i->kli',M[2],U,s)
    # Update F
    for i in range(nops):
        F[i][3] = np.einsum('bxc,be,eaf,cf->xa',np.conj(M[3]),ops[i][3],M[3],F[i][4])
    # NEXT SITE
    print('\tSite 2')
    for i in range(nops):
        if i == 0:
            H = np.einsum('jp,in,kq->ijknpq',F[i][2],ops[i][2],F[i][3])
        else:
            H += np.einsum('jp,in,kq->ijknpq',F[i][2],ops[i][2],F[i][3])
    H = np.reshape(H,(16,16))
    u,v = np.linalg.eig(H)
    # select max eigenvalue
    max_ind = np.argsort(u)[-1]
    E = u[max_ind]
    v = v[:,max_ind]
    print('\t\tCurrent Energy = {}'.format(E))
    M[2] = np.reshape(v,(2,4,2))
    # Right Normalize 
    M_reshape = np.swapaxes(M[2],0,1)
    M_reshape = np.reshape(M_reshape,(4,4))
    (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
    M_reshape = np.reshape(V,(4,2,2))
    M[2] = np.swapaxes(M_reshape,0,1)
    M[1] = np.einsum('klj,ji,i->kli',M[1],U,s)
    # Update F
    for i in range(nops):
        F[i][2] = np.einsum('bxc,be,eaf,cf->xa',np.conj(M[2]),ops[i][2],M[2],F[i][3])
    # NEXT SITE
    print('\tSite 1')
    for i in range(nops):
        if i == 0:
            H = np.einsum('jp,in,kq->ijknpq',F[i][1],ops[i][1],F[i][2])
        else:
            H += np.einsum('jp,in,kq->ijknpq',F[i][1],ops[i][1],F[i][2])
    H = np.reshape(H,(16,16))
    u,v = np.linalg.eig(H)
    # select max eigenvalue
    max_ind = np.argsort(u)[-1]
    E = u[max_ind]
    v = v[:,max_ind]
    print('\t\tCurrent Energy = {}'.format(E))
    M[1] = np.reshape(v,(2,2,4))
    # Right Normalize 
    M_reshape = np.swapaxes(M[1],0,1)
    M_reshape = np.reshape(M_reshape,(2,8))
    (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
    M_reshape = np.reshape(V,(2,2,4))
    M[1] = np.swapaxes(M_reshape,0,1)
    M[0] = np.einsum('klj,ji,i->kli',M[0],U,s)
    # Update F
    for i in range(nops):
        F[i][1] = np.einsum('bxc,be,eaf,cf->xa',np.conj(M[1]),ops[i][1],M[1],F[i][2])
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
