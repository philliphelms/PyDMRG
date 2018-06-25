import numpy as np
import scipy.linalg as la

#print('\n\n')
print('*'*76)
print('Beginning Two Site Calculations')

######## Inputs ##############################
# Model
N=2
alpha = 0.35  # In at left
beta = 2/3    # Exit at right
s = 0.        # Exponential weighting
gamma = 0     # Exit at left
delta = 0     # In at right
p = 1         # Jump right
q = 0         # Jump Left
# Optimization
tol = 1e-10
maxIter = 10
##############################################

######## Prereqs #############################
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

# Calculate Hamiltonian ######################
H = np.zeros((2**N,2**N))
occ = np.zeros((2**N,N),dtype=int)
sum_occ = np.zeros(2**N)
for i in range(2**N):
    occ[i,:] = np.asarray(list(map(lambda x: int(x),'0'*(N-len(bin(i)[2:]))+bin(i)[2:])))
    sum_occ[i] = np.sum(occ[i,:])
# Calculate Hamiltonian Methods
for i in range(2**N):
    i_occ = occ[i,:]
    for j in range(2**N):
        j_occ = occ[j,:]
        tmp_mat = np.array([[1]])
        for k in range(N):
            tmp_mat = np.einsum('ij,jk->ik',tmp_mat,W[k][:,:,i_occ[k],j_occ[k]])
        H[i,j] += tmp_mat[[0]]
# Diagonalize Ham
e,lwf_ed,rwf_ed = la.eig(H,left=True)
inds = np.argsort(e)
lwf_ed =lwf_ed[:,inds[-1]]
rwf_ed = rwf_ed[:,inds[-1]]
lwf_ed = lwf_ed/np.sum(lwf_ed*rwf_ed)
print('Original Wavefunction {}'.format(rwf_ed))
##############################################


######## Prereqs #############################
# Create MPS
M = []
M.insert(len(M),np.ones((2,1,2)))
M.insert(len(M),np.ones((2,2,1)))
Ml = []
Ml.insert(len(Ml),np.ones((2,1,2)))
Ml.insert(len(Ml),np.ones((2,2,1)))
# Make MPS Right Canonical
M_reshape = np.swapaxes(M[1],0,1)
M_reshape = np.reshape(M_reshape,(2,2))
(U,s,V) = np.linalg.svd(M_reshape,full_matrices=True)
M_reshape = np.reshape(V,(2,2,1))
M[1] = np.swapaxes(M_reshape,0,1)
M[0] = np.einsum('klj,ji,i->kli',M[0],U,s)
Ml_reshape = np.swapaxes(Ml[1],0,1)
Ml_reshape = np.reshape(Ml_reshape,(2,2))
(Ul,sl,Vl) = np.linalg.svd(Ml_reshape,full_matrices=True)
Ml_reshape = np.reshape(Vl,(2,2,1))
Ml[1] = np.swapaxes(Ml_reshape,0,1)
Ml[0] = np.einsum('klj,ji,i->kli',Ml[0],Ul,sl)
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
    print('\tSite 0')
    H = np.einsum('ijk,lmin,npq,mtru,stv->rkqusv',np.conj(M[0]),W[0],M[0],W[1],np.array([[[1]]]))
    H = np.einsum('ijk,jlmn,olp->mionkp',\
            np.einsum('ijk,lmio,opq->kmq',np.conj(M[0]),W[0],M[0]),\
            W[1],np.array([[[1]]]))
    H = np.reshape(H,(4,4))
    #u,v = np.linalg.eig(H)
    u,vl,vr = la.eig(H,left=True)
    # select max eigenvalue
    max_ind = np.argsort(u)[-1]
    E = u[max_ind]
    vr = vr[:,max_ind]
    vl = vl[:,max_ind]
    print('\t\tCurrent Energy = {}'.format(E))
    M[1] = np.reshape(vr,(2,2,1)) # Could this be wrong?!?!
    Ml[1]= np.reshape(vl,(2,2,1))
    # Right Normalize
    print('\t\tCheck Energy = {}'.format(np.einsum('ijk,lmin,npq,rks,mtru,uqv->',np.conj(M[0]),W[0],M[0],np.conj(M[1]),W[1],M[1])))
    M_reshape = np.swapaxes(M[1],0,1)
    M_reshape = np.reshape(M_reshape,(2,2))
    (U,s,V) = np.linalg.svd(M_reshape,full_matrices=True)
    M_reshape = np.reshape(V,(2,2,1))
    M[1] = np.swapaxes(M_reshape,0,1)
    M[0] = np.einsum('klj,ji,i->kli',M[0],U,s)
    Ml_reshape = np.swapaxes(Ml[1],0,1)
    Ml_reshape = np.reshape(Ml_reshape,(2,2))
    (Ul,sl,Vl) = np.linalg.svd(Ml_reshape,full_matrices=True)
    Ml_reshape = np.reshape(Vl,(2,2,1))
    Ml[1] = np.swapaxes(Ml_reshape,0,1)
    Ml[0] = np.einsum('klj,ji,i->kli',Ml[0],Ul,sl)
    print('\t\tCheck Energy = {}'.format(np.einsum('ijk,lmin,npq,rks,mtru,uqv->',np.conj(M[0]),W[0],M[0],np.conj(M[1]),W[1],M[1])))
# Left Sweep -----------------------------
    # Optimization
    print('\tSite 1')
    H = np.einsum('ijk,jlmn,olp->mionkp',np.array([[[1]]]),W[0],\
            np.einsum('ijk,lmio,opq,kmq->jlp',np.conj(M[1]),W[1],M[1],np.array([[[1]]])))
    H = np.reshape(H,(4,4))
    #u,v = np.linalg.eig(H)
    u,vl,vr = la.eig(H,left=True)
    # select max eigenvalue
    max_ind = np.argsort(u)[-1]
    E = u[max_ind]
    vr = vr[:,max_ind]
    vl = vl[:,max_ind]
    print('\t\tCurrent Energy = {}'.format(E))
    M[0] = np.reshape(vr,(2,1,2)) # Could this be wrong?!?!
    Ml[0]= np.reshape(vl,(2,1,2))
    # Left Normalize
    print('\t\tCheck Energy = {}'.format(np.einsum('ijk,lmin,npq,rks,mtru,uqv->',np.conj(M[0]),W[0],M[0],np.conj(M[1]),W[1],M[1])))
    M_reshape = np.reshape(M[0],(2,2))
    (U,s,V) = np.linalg.svd(M_reshape,full_matrices=True)
    M[0] = np.reshape(U,(2,1,2))
    M[1] = np.einsum('i,ij,kjl->kil',s,V,M[1])
    Ml_reshape = np.reshape(Ml[0],(2,2))
    (Ul,sl,Vl) = np.linalg.svd(Ml_reshape,full_matrices=True)
    Ml[0] = np.reshape(Ul,(2,1,2))
    Ml[1] = np.einsum('i,ij,kjl->kil',sl,Vl,Ml[1])
    print('\t\tCheck Energy = {}'.format(np.einsum('ijk,lmin,npq,rks,mtru,uqv->',np.conj(M[0]),W[0],M[0],np.conj(M[1]),W[1],M[1])))
# Convergence Test -----------------------
    if np.abs(E-E_prev) < tol:
        print('-'*50+'\nConverged at E = {}'.format(E)+'\n'+'-'*50)
        converged = True
    elif iterCnt > maxIter:
        print('Convergence not acheived')
        converged = True
    else:
        iterCnt += 1
        E_prev = np.einsum('ijk,lmin,npq,rks,mtru,uqv->',np.conj(M[0]),W[0],M[0],np.conj(M[1]),W[1],M[1])
##############################################

# Calculate State ############################
occ = np.zeros((2**N,N),dtype=int)
sum_occ = np.zeros(2**N)
for i in range(2**N):
    occ[i,:] = np.asarray(list(map(lambda x: int(x),'0'*(N-len(bin(i)[2:]))+bin(i)[2:])))
    sum_occ[i] = np.sum(occ[i,:])
# Calculate Wavefunction
rwf_dmrg = np.zeros(2**N,dtype=np.complex128)
lwf_dmrg = np.zeros(2**N,dtype=np.complex128)
for i in range(2**N):
    i_occ = occ[i,:]
    tmp_mat = np.array([[1]])
    tmp_matl = np.array([[1]])
    for k in range(N):
        tmp_mat  = np.einsum('ij,jk->ik',tmp_mat, M [k][i_occ[k],:,:])
        tmp_matl = np.einsum('ij,jk->ik',tmp_matl,Ml[k][i_occ[k],:,:])
    rwf_dmrg[i] = tmp_mat [0,0]
    lwf_dmrg[i] = tmp_matl[0,0]
##############################################

# Print Wavefunctions ########################
#print('\n\n')
#print('Right Wavefunctions')
#print('---------------------------------------')
#print('ED\tDMRG\tDiff')
#for i in range(2**N):
#    print('{}\t{}\t{}'.format(np.real(np.round(rwf_ed[i],3)),np.real(np.round(rwf_dmrg[i],3)),np.real(np.abs(rwf_ed[i])-np.abs(rwf_dmrg[i]))))
#print('\n\n')
#print('Left Wavefunctions')
#print('---------------------------------------')
#print('ED\tDMRG\tDiff')
#for i in range(2**N):
#    print('{}\t{}\t{}'.format(np.real(np.round(lwf_ed[i],3)),np.real(np.round(lwf_dmrg[i],3)),np.real(np.abs(lwf_ed[i])-np.abs(lwf_dmrg[i]))))
#print('\n\n')
#print('DMRG')
#print(np.sum(rwf_dmrg*rwf_dmrg))
#print(np.sum(lwf_dmrg*lwf_dmrg))
#print(np.sum(lwf_dmrg*rwf_dmrg))
#print('ED')
#print(np.sum(rwf_ed*rwf_ed))
#print(np.sum(lwf_ed*lwf_ed))
#print(np.sum(lwf_ed*rwf_ed))
##############################################

# Decompose Resulting Exact WF ###############
M_analytic = []
M_analytic.insert(len(M_analytic),np.ones((2,1,2)))
M_analytic.insert(len(M_analytic),np.ones((2,2,1)))
psi = np.zeros((2,2))
psi[0,0] = rwf_ed[0]
psi[0,1] = rwf_ed[1]
psi[1,0] = rwf_ed[2]
psi[1,1] = rwf_ed[3]
(U,s,V) = np.linalg.svd(psi,full_matrices=True)
print(V.shape)
print(U.shape)
print(s.shape)
M_analytic[1] = np.reshape(V,(2,2,1))
M_analytic[0] = np.einsum('ij,j->ij',U,s)
##############################################
print(M_analytic[1])
print(M[1])
print(M_analytic[0])
print(M[0])

# Check that state hasn't changed ###########
occ = np.zeros((2**N,N),dtype=int)
sum_occ = np.zeros(2**N)
for i in range(2**N):
    occ[i,:] = np.asarray(list(map(lambda x: int(x),'0'*(N-len(bin(i)[2:]))+bin(i)[2:])))
    sum_occ[i] = np.sum(occ[i,:])
# Calculate Wavefunction
rwf_chk = np.zeros(2**N,dtype=np.complex128)
#lwf_dmrg = np.zeros(2**N,dtype=np.complex128)
for i in range(2**N):
    i_occ = occ[i,:]
    tmp_mat = np.array([[1]])
    #tmp_matl = np.array([[1]])
    for k in range(N):
        tmp_mat  = np.einsum('ij,jk->ik',tmp_mat,M_analytic[k][i_occ[k],:,:])
        #tmp_matl = np.einsum('ij,jk->ik',tmp_matl,Ml[k][i_occ[k],:,:])
    rwf_chk[i] = tmp_mat [0,0]
    #lwf_dmrg[i] = tmp_matl[0,0]
##############################################
