import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
#from pyscf import lib # Library containing davidson algorithm

######## Inputs ##############################
# SEP Model
N = 10
alpha = 0.35     # In at left
beta = 2/3       # Exit at right
s = -1           # Exponential weighting
p = 1            # Jump right
target_state = 3 # The targeted excited state
# Optimization
tol = 1e-5
maxIter = 10
maxBondDim = 16
##############################################

######## Prereqs #############################
# Create MPS
M = []
M.insert(len(M),np.ones((2,1,maxBondDim)))
for i in range(N):
    M.insert(len(M),np.ones((2,maxBondDim,maxBondDim)))
M.insert(len(M),np.ones((2,maxBondDim,1)))
# Create MPO
Sp = np.array([[0,1],[0,0]])
Sm = np.array([[0,0],[1,0]])
n = np.array([[0,0],[0,1]])
v = np.array([[1,0],[0,0]])
I = np.array([[1,0],[0,1]])
z = np.array([[0,0],[0,0]])
W = []
W.insert(len(W),np.array([[alpha*(np.exp(-s)*Sm-v),np.exp(-s)*Sp,-n,I]]))
for i in range(N-2):
    W.insert(len(W),np.array([[I,z,z,z],[Sm,z,z,z],[v,z,z,z],[z,np.exp(-s)*Sp,-n,I]]))
W.insert(len(W),np.array([[I],[Sm],[v],[beta*(np.exp(-s)*Sp-n)]]))
# Create F
F = []
F.insert(len(F),np.array([[[1]]]))
for i in range(N):
    F.insert(len(F),np.zeros((maxBondDim,4,maxBondDim)))
F.insert(len(F),np.array([[[1]]]))
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
for i in range(int(N)-1,0,-1):
    F[i] = np.einsum('bxc,ydbe,eaf,cdf->xya',np.conj(M[i]),W[i],M[i],F[i+1])
##############################################

# Optimization Sweeps ########################
converged = False
iterCnt = 0
E_prev = 0
while not converged:
# Right Sweep ----------------------------
    print('Right Sweep {}'.format(iterCnt))
    for i in range(N-1):
        (n1,n2,n3) = M[i].shape
        def opt_fun(x): # Function to be called by Davidson algorithm
            x_reshape = np.reshape(x,M[i].shape)
            in_sum1 = np.einsum('ijk,lmk->ijlm',F[i+1],x_reshape)
            in_sum2 = np.einsum('njol,ijlm->noim',W[i],in_sum1)
            fin_sum = np.einsum('pnm,noim->opi',F[i],in_sum2)
            print(fin_sum.shape)
            return np.reshape(fin_sum,-1)
        # Cheating way to get size
        H = np.einsum('jlp,lmin,kmq->ijknpq',F[i],W[i],F[i+1])
        (n1,n2,n3,n4,n5,n6) = H.shape
        opt_lin_op = LinearOperator((n1*n2*n3,n4*n5*n6),matvec=opt_fun)
        #init_guess = [np.reshape(M[i],-1)]*(target_state+1)
        u,v = eigs(opt_lin_op,k=min(target_state+1,n1*n2*n3-2),which='LR')
        print(v.shape)
        # select max eigenvalue
        sort_inds = np.argsort(np.real(u))[::-1]
        try:
            E = u[sort_inds[min(target_state,len(u)-1)]]
            v = v[:,sort_inds[min(target_state,len(u)-1)]]
        except:
            E = u
            v = v
        print(v.shape)
        print('\tEnergy at site {} = {}'.format(i,E))
        M[i] = np.reshape(v,(n1,n2,n3))
        print(M[i].shape)
        # Right Normalize
        M_reshape = np.reshape(M[i],(n1*n2,n3))
        print(M_reshape.shape)
        (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
        print(U.shape)
        print(s.shape)
        print(V.shape)
        M[i] = np.reshape(U,(n1,n2,n3))
        M[i+1] = np.einsum('i,ij,kjl->kil',s,V,M[i+1])
        # Update F
        F[i+1] = np.einsum('jlp,ijk,lmin,npq->kmq',F[i],np.conj(M[i]),W[i],M[i])
# Left Sweep -----------------------------
    print('Left Sweep {}'.format(iterCnt))
    for i in range(N-1,0,-1):
        (n1,n2,n3) = M[i].shape
        def opt_fun(x): # Function to be called by Davidson algorithm
            x_reshape = np.reshape(x,M[i].shape)
            in_sum1 = np.einsum('ijk,lmk->ijlm',F[i+1],x_reshape)
            in_sum2 = np.einsum('njol,ijlm->noim',W[i],in_sum1)
            fin_sum = np.einsum('pnm,noim->opi',F[i],in_sum2)
            return np.reshape(fin_sum,-1)
        # Cheating way to get size
        H = np.einsum('jlp,lmin,kmq->ijknpq',F[i],W[i],F[i+1])
        (n1,n2,n3,n4,n5,n6) = H.shape
        opt_lin_op = LinearOperator((n1*n2*n3,n4*n5*n6),matvec=opt_fun)
        #init_guess = [np.reshape(M[i],-1)]*(target_state+1)
        u,v = eigs(opt_lin_op,k=min(target_state+1,n1*n2*n3-2),which='LR')
        # select max eigenvalue
        sort_inds = np.argsort(np.real(u))[::-1]
        try:
            E = u[sort_inds[min(target_state,len(u)-1)]]
            v = v[:,sort_inds[min(target_state,len(u)-1)]]
        except:
            E = u
            v = v
        print('\tEnergy at site {} = {}'.format(i,E))
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
