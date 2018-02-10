import numpy as np

######## Inputs ##############################
# SEP Model
N = 10
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
maxBondDim = 8
##############################################

######## Prereqs #############################
# Create MPS
M = []
for i in range(int(N/2)):
    M.insert(len(M),np.ones((2,min(2**(i),maxBondDim),min(2**(i+1),maxBondDim))))
for i in range(int(N/2))[::-1]:
    M.insert(len(M),np.ones((2,min(2**(i+1),maxBondDim),min(2**i,maxBondDim))))
# Create basic operators
Sp = np.array([[0,1],[0,0]])
Sm = np.array([[0,0],[1,0]])
n = np.array([[0,0],[0,1]])
v = np.array([[1,0],[0,0]])
I = np.array([[1,0],[0,1]])
z = np.array([[0,0],[0,0]])
# Possible Operators
# In at left
# Hopping to the right 1
# Hopping to the right 2
# Out at right
ops = []
# In at left operator
tmp_op = []
tmp_op.insert(len(tmp_op),alpha*(np.exp(-s)*Sm-v))
for i in range(N-1):
    tmp_op.insert(len(tmp_op),I)
# Hopping to the right 1
for i in range(N-1):
    tmp_op1 = []
    tmp_op2 = []
    for j in range(N):
        if i == j:
            tmp_op1.insert(len(tmp_op1),np.exp(-s)*Sp)
            tmp_op2.insert(len(tmp_op2),-n)
        if i+1 == j:
            tmp_op1.insert(len(tmp_op1),Sm)
            tmp_op2.insert(len(tmp_op2),v)
        else:
            tmp_op1.insert(len(tmp_op1),I)
            tmp_op2.insert(len(tmp_op2),I)
    ops.insert(len(ops),tmp_op1)
    ops.insert(len(ops),tmp_op2)
# Out at right operator
tmp_op = []
for i in range(N-1):
    tmp_op.insert(len(tmp_op),I)
tmp_op.insert(len(tmp_op),beta*(np.exp(-s)*Sp-n))

nops = len(ops)
# Create F
F = []
for i in range(nops):
    F_tmp = []
    F_tmp.insert(len(F_tmp),np.array([[1]]))
    for j in range(int(N/2)):
        F_tmp.insert(len(F_tmp),np.zeros((min(2**(j+1),maxBondDim),min(2**(j+1),maxBondDim))))
    for j in range(int(N/2)-1,0,-1):
        F_tmp.insert(len(F_tmp),np.zeros((min(2**(j),maxBondDim),min(2**j,maxBondDim))))
    F_tmp.insert(len(F_tmp),np.array([[1]]))
    F.insert(len(F),F_tmp)
##############################################

# Make MPS Right Canonical ###################
for i in range(int(N)-1,0,-1):
    M_reshape = np.swapaxes(M[i],0,1)
    (n1,n2,n3) = M_reshape.shape
    M_reshape = np.reshape(M_reshape,(n1,n2*n3))
    (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
    M_reshape = np.reshape(V,(n1,n2,n3))
    M[i] = np.swapaxes(M_reshape,0,1)
    M[i-1] = np.einsum('klj,ji,i->kli',M[i-1],U,s)
##############################################

# Calculate Initial F ########################
for j in range(nops):
    for i in range(int(N)-1,0,-1):
        F[j][i] = np.einsum('bxc,be,eaf,cf->xa',np.conj(M[i]),ops[j][i],M[i],F[j][i+1])
##############################################

# Optimization Sweeps ########################
converged = False
iterCnt = 0
E_prev = 0
while not converged:
# Right Sweep ----------------------------
    print('Right Sweep {}'.format(iterCnt))
    for i in range(N-1):
        for j in range(nops):
            if j == 0:
                H = np.einsum('jp,in,kq->ijknpq',F[j][i],ops[j][i],F[j][i+1])
            else:
                H += np.einsum('jp,in,kq->ijknpq',F[j][i],ops[j][i],F[j][i+1])
        (n1,n2,n3,n4,n5,n6) = H.shape
        H = np.reshape(H,(n1*n2*n3,n4*n5*n6))
        print(H)
        u,v = np.linalg.eig(H)
        # select max eigenvalue
        max_ind = np.argsort(u)[-1]
        E = u[max_ind]
        v = v[:,max_ind]
        print('\tEnergy at site {}= {}'.format(i,E))
        M[i] = np.reshape(v,(n1,n2,n3))
        # Right Normalize
        M_reshape = np.reshape(M[i],(n1*n2,n3))
        (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
        M[i] = np.reshape(U,(n1,n2,n3))
        M[i+1] = np.einsum('i,ij,kjl->kil',s,V,M[i+1])
        # Update F
        for j in range(nops):
            F[j][i+1] = np.einsum('jp,ijk,in,npq->kq',F[j][i],np.conj(M[i]),ops[j][i],M[i])
# Left Sweep -----------------------------
    print('Left Sweep {}'.format(iterCnt))
    for i in range(N-1,0,-1):
        H = np.einsum('jlp,lmin,kmq->ijknpq',F[i],W[i],F[i+1])
        (n1,n2,n3,n4,n5,n6) = H.shape
        H = np.reshape(H,(n1*n2*n3,n4*n5*n6))
        u,v = np.linalg.eig(H)
        # select max eigenvalue
        max_ind = np.argsort(u)[-1]
        E = u[max_ind]
        v = v[:,max_ind]
        print('\tEnergy at site {}= {}'.format(i,E))
        M[i] = np.reshape(v,(n1,n2,n3))
        # Right Normalize 
        M_reshape = np.swapaxes(M[i],0,1)
        M_reshape = np.reshape(M_reshape,(n2,n1*n3))
        (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
        M_reshape = np.reshape(V,(n2,n1,n3))
        M[i] = np.swapaxes(M_reshape,0,1)
        M[i-1] = np.einsum('klj,ji,i->kli',M[i-1],U,s)
        # Update F
        F[i] = np.einsum('bxc,ydbe,eaf,cdf->xya',np.conj(M[i]),W[i],M[i],F[i+1])
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
