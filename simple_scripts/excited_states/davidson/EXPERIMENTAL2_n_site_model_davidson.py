import numpy as np
from pyscf import lib # Library containing davidson algorithm

######## Inputs ##############################
# SEP Model
N = 10
alpha = 0.35     # In at left
beta = 2/3       # Exit at right
s = -1           # Exponential weighting
p = 1            # Jump right
target_state = 1 # The targeted excited state
# Optimization
tol = 1e-5
maxIter = 10
maxBondDim = 16
##############################################

######## Prereqs #############################
# Create MPS
M = []
for i in range(target_state+1):
    M_inner = []
    for i in range(int(N/2)):
        M_inner.insert(len(M_inner),np.ones((2,min(2**(i),maxBondDim),min(2**(i+1),maxBondDim))))
    for i in range(int(N/2))[::-1]:
        M_inner.insert(len(M_inner),np.ones((2,min(2**(i+1),maxBondDim),min(2**i,maxBondDim))))
    M.insert(len(M),M_inner)
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
for i in range(target_state+1):
    F_inner = []
    F_inner.insert(len(F_inner),np.array([[[1]]]))
    for i in range(int(N/2)):
        F_inner.insert(len(F_inner),np.zeros((min(2**(i+1),maxBondDim),4,min(2**(i+1),maxBondDim))))
    for i in range(int(N/2)-1,0,-1):
        F_inner.insert(len(F_inner),np.zeros((min(2**(i),maxBondDim),4,min(2**i,maxBondDim))))
    F_inner.insert(len(F_inner),np.array([[[1]]]))
    F.insert(len(F),F_inner)
##############################################

# Make MPS Right Canonical ###################
for i in range(int(N)-1,0,-1):
    for j in range(target_state+1):
        M_reshape = np.swapaxes(M[j][i],0,1)
        (n1,n2,n3) = M_reshape.shape
        M_reshape = np.reshape(M_reshape,(n1,n2*n3))
        (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
        M_reshape = np.reshape(V,(n1,n2,n3))
        M[j][i] = np.swapaxes(M_reshape,0,1)
        M[j][i-1] = np.einsum('klj,ji,i->kli',M[j][i-1],U,s)
##############################################

# Calculate Initial F ########################
for i in range(int(N)-1,0,-1):
    for j in range(target_state+1):
        F[j][i] = np.einsum('bxc,ydbe,eaf,cdf->xya',np.conj(M[j][i]),W[i],M[j][i],F[j][i+1])
##############################################

# Optimization Sweeps ########################
converged = False
iterCnt = 0
E_prev = 0
while not converged:
# Right Sweep ----------------------------
    print('Right Sweep {}'.format(iterCnt))
    for i in range(N-1):
        (n1,n2,n3) = M[0][i].shape
        def opt_fun(x): # Function to be called by Davidson algorithm
            x_reshape = np.reshape(x,(n1,n2,n3))
            in_sum1 = np.einsum('ijk,lmk->ijlm',F[target_state][i+1],x_reshape) # HOW DO YOU PICK WHICH F TO WORK WITH???
            in_sum2 = np.einsum('njol,ijlm->noim',W[i],in_sum1)
            fin_sum = np.einsum('pnm,noim->opi',F[target_state][i],in_sum2)
            return -np.reshape(fin_sum,-1)
        def precond(dx,e,x0): # A second dummy algorithm
            return dx
        init_guess = []
        for j in range(target_state+1):
            init_guess.insert(len(init_guess),np.reshape(M[j][i],-1))
        u,v = lib.eig(opt_fun,init_guess,precond,nroots=target_state+1)
        print(u)
        # select max eigenvalue
        sort_inds = np.argsort(np.real(u))#[::-1]
        try:
            E = -u[sort_inds[min(target_state,len(u)-1)]]
            #v = v[sort_inds[min(target_state,len(u)-1)]]
        except:
            E = -u
            #v = v
        print('\tEnergy at site {} = {}'.format(i,E))
        M[target_state][i] = np.reshape(v[sort_inds[min(target_state,len(u)-1)]],(n1,n2,n3))
        # Right Normalize
        M_reshape = np.reshape(M[target_state][i],(n1*n2,n3))
        (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
        M[target_state][i] = np.reshape(U,(n1,n2,n3))
        M[target_state][i+1] = np.einsum('i,ij,kjl->kil',s,V,M[target_state][i+1])
        # Update F
        F[target_state][i+1] = np.einsum('jlp,ijk,lmin,npq->kmq',F[target_state][i],np.conj(M[target_state][i]),W[i],M[target_state][i])
# Left Sweep -----------------------------
    print('Left Sweep {}'.format(iterCnt))
    for i in range(N-1,0,-1):
        (n1,n2,n3) = M[0][i].shape
        def opt_fun(x): # Function to be called by Davidson algorithm
            x_reshape = np.reshape(x,(n1,n2,n3))
            in_sum1 = np.einsum('ijk,lmk->ijlm',F[0][i+1],x_reshape)
            in_sum2 = np.einsum('njol,ijlm->noim',W[i],in_sum1)
            fin_sum = np.einsum('pnm,noim->opi',F[0][i],in_sum2)
            return -np.reshape(fin_sum,-1)
        def precond(dx,e,x0): # A second dummy algorithm
            return dx
        init_guess = []
        for j in range(target_state+1):
            init_guess.insert(len(init_guess),np.reshape(M[j][i],-1))
        u,v = lib.eig(opt_fun,init_guess,precond,nroots=target_state+1)
        print(u)
        # select max eigenvalue
        sort_inds = np.argsort(np.real(u))#[::-1]
        try:
            E = -u[sort_inds[min(target_state,len(u)-1)]]
            #v = v[sort_inds[min(target_state,len(u)-1)]]
        except:
            E = -u
            #v = v
        print('\tEnergy at site {} = {}'.format(i,E))
        M[target_state][i] = np.reshape(v[target_state],(n1,n2,n3))
        # Right Normalize 
        M_reshape = np.swapaxes(M[target_state][i],0,1)
        M_reshape = np.reshape(M_reshape,(n2,n1*n3))
        (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
        M_reshape = np.reshape(V,(n2,n1,n3))
        M[target_state][i] = np.swapaxes(M_reshape,0,1)
        M[target_state][i-1] = np.einsum('kli,ji,i->kli',M[target_state][i-1],U,s)
        # Update F
        F[target_state][i] = np.einsum('bxc,ydbe,eaf,cdf->xya',np.conj(M[target_state][i]),W[i],M[target_state][i],F[target_state][i+1])
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
