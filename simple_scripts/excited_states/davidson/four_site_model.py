import numpy as np
from pyscf import lib


######## Inputs ##############################
# Model
alpha = 0.35     # In at left
beta = 2/3       # Exit at right
s = -1           # Exponential weighting
gamma = 0        # Exit at left
delta = 0        # In at right
p = 1            # Jump right
q = 0            # Jump Left
target_state = 0 # The targeted excited state
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
W = []
W.insert(len(W),np.array([[alpha*(np.exp(-s)*Sm-v),np.exp(-s)*Sp,-n,I]]))
W.insert(len(W),np.array([[I,z,z,z],[Sm,z,z,z],[v,z,z,z],[z,np.exp(-s)*Sp,-n,I]]))
W.insert(len(W),np.array([[I,z,z,z],[Sm,z,z,z],[v,z,z,z],[z,np.exp(-s)*Sp,-n,I]]))
W.insert(len(W),np.array([[I],[Sm],[v],[beta*(np.exp(-s)*Sp-n)]]))
# Create F
F = []
F.insert(len(F),np.array([[[1]]]))
F.insert(len(F),np.zeros((2,4,2)))
F.insert(len(F),np.zeros((4,4,4)))
F.insert(len(F),np.zeros((2,4,2)))
F.insert(len(F),np.array([[[1]]]))
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
F[3] = np.einsum('bxc,ydbe,eaf,cdf->xya',np.conj(M[3]),W[3],M[3],F[4])
F[2] = np.einsum('wsx,tywz,zva,xya->stv',np.conj(M[2]),W[2],M[2],F[3])
F[1] = np.einsum('rks,mtru,uqv,stv->kmq',np.conj(M[1]),W[1],M[1],F[2])
print(F[0])
##############################################

# Optimization Sweeps ########################
converged = False
iterCnt = 0
#E = np.einsum('ijk,lmin,npq,rks,mtru,uqv->jlpstv',np.conj(M[0]),W[0],M[0],np.conj(M[1]),W[1],M[1])
E_prev= np.einsum('jlp,ijk,lmin,npq,rks,mtru,uqv,wsx,tywz,zva,bxc,ydbe,eaf,cdf->',\
                  np.array([[[1]]]),\
                  np.conj(M[0]),W[0],M[0],\
                  np.conj(M[1]),W[1],M[1],\
                  np.conj(M[2]),W[2],M[2],\
                  np.conj(M[3]),W[3],M[3],\
                  np.array([[[1]]]))
while not converged:
# Right Sweep ----------------------------
    print('Right Sweep {}'.format(iterCnt))
    # Optimization
    print('\tSite 0')
    i = 0
    (n1,n2,n3) = M[i].shape
    def opt_fun(x): # Function to be called by Davidson algorithm
        x_reshape = np.reshape(x,M[i].shape)
        in_sum1 = np.einsum('ijk,lmk->ijlm',F[i+1],x_reshape)
        in_sum2 = np.einsum('njol,ijlm->noim',W[i],in_sum1)
        fin_sum = np.einsum('pnm,noim->opi',F[i],in_sum2)
        return np.reshape(fin_sum,-1)
    def precond(dx,e,x0): # A second dummy algorithm
        return dx
    init_guess = np.reshape(M[i],-1)
    u,v = lib.eig(opt_fun,init_guess,precond,nroots=9)
    # select max eigenvalue
    sort_inds = np.argsort(np.real(u))[::-1]
    print(u[sort_inds])
    E = u[sort_inds[target_state]]
    v = v[sort_inds[target_state]]
    print('\t\tCurrent Energy = {}'.format(E))
    M[0] = np.reshape(v,(2,1,2))
    # Right Normalize
    M_reshape = np.reshape(M[0],(2,2))
    (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
    M[0] = np.reshape(U,(2,1,2))
    M[1] = np.einsum('i,ij,kjl->kil',s,V,M[1])
    # Update F
    F[1] = np.einsum('jlp,ijk,lmin,npq->kmq',F[0],np.conj(M[0]),W[0],M[0])
    # NEXT SITE
    print('\tSite 1')
    i = 1
    (n1,n2,n3) = M[i].shape
    def opt_fun(x): # Function to be called by Davidson algorithm
        x_reshape = np.reshape(x,M[i].shape)
        in_sum1 = np.einsum('ijk,lmk->ijlm',F[i+1],x_reshape)
        in_sum2 = np.einsum('njol,ijlm->noim',W[i],in_sum1)
        fin_sum = np.einsum('pnm,noim->opi',F[i],in_sum2)
        return np.reshape(fin_sum,-1)
    def precond(dx,e,x0): # A second dummy algorithm
        return dx
    init_guess = np.reshape(M[i],-1)
    u,v = lib.eig(opt_fun,init_guess,precond,nroots=9)
    # select max eigenvalue
    sort_inds = np.argsort(np.real(u))[::-1]
    print(u[sort_inds])
    E = u[sort_inds[target_state]]
    v = v[sort_inds[target_state]]
    print('\t\tCurrent Energy = {}'.format(E))
    M[1] = np.reshape(v,(2,2,4))
    # Right Normalize
    M_reshape = np.reshape(M[1],(4,4))
    (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
    M[1] = np.reshape(U,(2,2,4))
    M[2] = np.einsum('i,ij,kjl->kil',s,V,M[2])
    # Update F
    F[2] = np.einsum('jlp,ijk,lmin,npq->kmq',F[1],np.conj(M[1]),W[1],M[1])
    # NEXT SITE
    print('\tSite 2')
    i = 2
    (n1,n2,n3) = M[i].shape
    def opt_fun(x): # Function to be called by Davidson algorithm
        x_reshape = np.reshape(x,M[i].shape)
        in_sum1 = np.einsum('ijk,lmk->ijlm',F[i+1],x_reshape)
        in_sum2 = np.einsum('njol,ijlm->noim',W[i],in_sum1)
        fin_sum = np.einsum('pnm,noim->opi',F[i],in_sum2)
        return np.reshape(fin_sum,-1)
    def precond(dx,e,x0): # A second dummy algorithm
        return dx
    init_guess = np.reshape(M[i],-1)
    u,v = lib.eig(opt_fun,init_guess,precond,nroots=9)
    # select max eigenvalue
    sort_inds = np.argsort(np.real(u))[::-1]
    print(u[sort_inds])
    E = u[sort_inds[target_state]]
    v = v[sort_inds[target_state]]
    print('\t\tCurrent Energy = {}'.format(E))
    M[2] = np.reshape(v,(2,4,2))
    # Right Normalize
    M_reshape = np.reshape(M[2],(8,2))
    (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
    M[2] = np.reshape(U,(2,4,2))
    M[3] = np.einsum('i,ij,kjl->kil',s,V,M[3])
    # Update F
    F[3] = np.einsum('jlp,ijk,lmin,npq->kmq',F[2],np.conj(M[2]),W[2],M[2])
# Left Sweep -----------------------------
    print('Left Sweep {}'.format(iterCnt))
    # Optimization
    print('\tSite 3')
    i = 3
    (n1,n2,n3) = M[i].shape
    def opt_fun(x): # Function to be called by Davidson algorithm
        x_reshape = np.reshape(x,M[i].shape)
        in_sum1 = np.einsum('ijk,lmk->ijlm',F[i+1],x_reshape)
        in_sum2 = np.einsum('njol,ijlm->noim',W[i],in_sum1)
        fin_sum = np.einsum('pnm,noim->opi',F[i],in_sum2)
        return np.reshape(fin_sum,-1)
    def precond(dx,e,x0): # A second dummy algorithm
        return dx
    init_guess = np.reshape(M[i],-1)
    u,v = lib.eig(opt_fun,init_guess,precond,nroots=9)
    # select max eigenvalue
    sort_inds = np.argsort(np.real(u))[::-1]
    print(u[sort_inds])
    E = u[sort_inds[target_state]]
    v = v[sort_inds[target_state]]
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
    F[3] = np.einsum('bxc,ydbe,eaf,cdf->xya',np.conj(M[3]),W[3],M[3],F[4])
    # NEXT SITE
    print('\tSite 2')
    i = 2
    (n1,n2,n3) = M[i].shape
    def opt_fun(x): # Function to be called by Davidson algorithm
        x_reshape = np.reshape(x,M[i].shape)
        in_sum1 = np.einsum('ijk,lmk->ijlm',F[i+1],x_reshape)
        in_sum2 = np.einsum('njol,ijlm->noim',W[i],in_sum1)
        fin_sum = np.einsum('pnm,noim->opi',F[i],in_sum2)
        return np.reshape(fin_sum,-1)
    def precond(dx,e,x0): # A second dummy algorithm
        return dx
    init_guess = np.reshape(M[i],-1)
    u,v = lib.eig(opt_fun,init_guess,precond,nroots=9)
    # select max eigenvalue
    sort_inds = np.argsort(np.real(u))[::-1]
    print(u[sort_inds])
    E = u[sort_inds[target_state]]
    v = v[sort_inds[target_state]]
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
    F[2] = np.einsum('bxc,ydbe,eaf,cdf->xya',np.conj(M[2]),W[2],M[2],F[3])
    # NEXT SITE
    print('\tSite 1')
    i = 1
    (n1,n2,n3) = M[i].shape
    def opt_fun(x): # Function to be called by Davidson algorithm
        x_reshape = np.reshape(x,M[i].shape)
        in_sum1 = np.einsum('ijk,lmk->ijlm',F[i+1],x_reshape)
        in_sum2 = np.einsum('njol,ijlm->noim',W[i],in_sum1)
        fin_sum = np.einsum('pnm,noim->opi',F[i],in_sum2)
        return np.reshape(fin_sum,-1)
    def precond(dx,e,x0): # A second dummy algorithm
        return dx
    init_guess = np.reshape(M[i],-1)
    u,v = lib.eig(opt_fun,init_guess,precond,nroots=9)
    # select max eigenvalue
    sort_inds = np.argsort(np.real(u))[::-1]
    print(u[sort_inds])
    E = u[sort_inds[target_state]]
    v = v[sort_inds[target_state]]
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
    F[1] = np.einsum('bxc,ydbe,eaf,cdf->xya',np.conj(M[1]),W[1],M[1],F[2])
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
