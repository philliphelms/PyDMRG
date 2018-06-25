import numpy as np
import scipy.linalg as la

#############################################################
# Goal: See if left eigenvectors can be determined via 
# some DMRG Procedure
#
# 1 - Calculate Full Hamiltonian of a TASEP System
# 2 - Diagonalize and Calculate Left and Right Eigenvectors
# 3 - Attempt a DMRG solution by either:
#     a - Doing Normal DMRG but saving Left eigenvectors
#     b - Inverting the MPO
#############################################################

######## Inputs ##############################
# SEP Model
N = 2
alpha = 0.5  # In at left
beta = 0.5    # Exit at right
bias = 1.        # Exponential weighting
gamma = 0.     # Exit at left
delta = 0.     # In at right
p = 1.         # Jump right
q = 0.         # Jump Left
# Optimization
tol = 1e-5
maxIter = 10
maxBondDim = 16
##############################################

######## Create MPO ##########################
Sp = np.array([[0,1],[0,0]])
Sm = np.array([[0,0],[1,0]])
n = np.array([[0,0],[0,1]])
v = np.array([[1,0],[0,0]])
I = np.array([[1,0],[0,1]])
z = np.array([[0,0],[0,0]])
V_op = v
W = []
W.insert(len(W),np.array([[alpha*(np.exp(-bias)*Sm-v),np.exp(-bias)*Sp,-n,I]]))
for i in range(N-2):
    W.insert(len(W),np.array([[I,z,z,z],[Sm,z,z,z],[v,z,z,z],[z,np.exp(-bias)*Sp,-n,I]]))
W.insert(len(W),np.array([[I],[Sm],[v],[beta*(np.exp(-bias)*Sp-n)]]))
##############################################

######## Calculate Hamiltonian  ##############
# Enumerate all possible states
H = np.zeros((2**N,2**N))
occ = np.zeros((2**N,N),dtype=int)
sum_occ = np.zeros(2**N)
for i in range(2**N):
    occ[i,:] = np.asarray(list(map(lambda x: int(x),'0'*(N-len(bin(i)[2:]))+bin(i)[2:])))
    sum_occ[i] = np.sum(occ[i,:])
# Sort Hamiltonian into blocks
if False:
    inds = np.argsort(sum_occ)
    sum_occ = sum_occ[inds]
    occ = occ[inds,:]
# Calculate Hamiltonian Methods
for i in range(2**N):
    i_occ = occ[i,:]
    for j in range(2**N):
        j_occ = occ[j,:]
        tmp_mat = np.array([[1]])
        for k in range(N):
            tmp_mat = np.einsum('ij,jk->ik',tmp_mat,W[k][:,:,i_occ[k],j_occ[k]])
        H[i,j] += tmp_mat[[0]]
##############################################

######## Diagonalize Ham #####################
e,lwf_ed,rwf_ed = la.eig(H,left=True)
inds = np.argsort(e)
lwf_ed =lwf_ed[:,inds[-1]]
rwf_ed = rwf_ed[:,inds[-1]]
lwf_ed = lwf_ed/np.sum(lwf_ed*rwf_ed)
print('Exact Diagonalization Energy = {}'.format(e[inds[-1]]))
##############################################

######## Setup TN  ###########################
# Create MPS
M = []
for i in range(int(N/2)):
    M.insert(len(M),np.ones((2,min(2**(i),maxBondDim),min(2**(i+1),maxBondDim))))
for i in range(int(N/2))[::-1]:
    M.insert(len(M),np.ones((2,min(2**(i+1),maxBondDim),min(2**i,maxBondDim))))
Ml = []
for i in range(int(N/2)):
    Ml.insert(len(Ml),np.ones((2,min(2**i,maxBondDim),min(2**(i+1),maxBondDim))))
for i in range(int(N/2))[::-1]:
    Ml.insert(len(Ml),np.ones((2,min(2**(i+1),maxBondDim),min(2**i,maxBondDim))))
# Create F
F = []
F.insert(len(F),np.array([[[1]]]))
for i in range(int(N/2)):
    F.insert(len(F),np.zeros((min(2**(i+1),maxBondDim),4,min(2**(i+1),maxBondDim))))
for i in range(int(N/2)-1,0,-1):
    F.insert(len(F),np.zeros((min(2**(i),maxBondDim),4,min(2**i,maxBondDim))))
F.insert(len(F),np.array([[[1]]]))
Fl = []
Fl.insert(len(Fl),np.array([[[1]]]))
for i in range(int(N/2)):
    Fl.insert(len(Fl),np.zeros((min(2**(i+1),maxBondDim),4,min(2**(i+1),maxBondDim))))
for i in range(int(N/2)-1,0,-1):
    Fl.insert(len(Fl),np.zeros((min(2**(i),maxBondDim),4,min(2**i,maxBondDim))))
Fl.insert(len(Fl),np.array([[[1]]]))
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
for i in range(int(N)-1,0,-1):
    M_reshape = np.swapaxes(Ml[i],0,1)
    (n1,n2,n3) = M_reshape.shape
    M_reshape = np.reshape(M_reshape,(n1,n2*n3))
    (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
    M_reshape = np.reshape(V,(n1,n2,n3))
    Ml[i] = np.swapaxes(M_reshape,0,1)
    Ml[i-1] = np.einsum('klj,ji,i->kli',M[i-1],U,s)
##############################################

# Calculate Initial F ########################
for i in range(int(N)-1,0,-1):
    F[i] = np.einsum('bxc,ydbe,eaf,cdf->xya',np.conj(M[i]),W[i],M[i],F[i+1])
for i in range(int(N)-1,0,-1):
    Fl[i] = np.einsum('bxc,ydbe,eaf,cdf->xya',np.conj(Ml[i]),W[i],Ml[i],Fl[i+1])
##############################################

# Optimization Sweeps ########################
converged = False
iterCnt = 0
E_prev = 0
rho = np.zeros(len(M))
while not converged:
# Right Sweep ----------------------------
    print('Right Sweep {}'.format(iterCnt))
    for i in range(N-1):
        H = np.einsum('jlp,lmin,kmq->ijknpq',F[i],W[i],F[i+1])
        (n1,n2,n3,n4,n5,n6) = H.shape
        H = np.reshape(H,(n1*n2*n3,n4*n5*n6))
        u,vl,vr = la.eig(H,left=True)
        # select max eigenvalue
        max_ind = np.argsort(u)[-1]
        E = u[max_ind]
        vr = vr[:,max_ind]
        vl = vl[:,max_ind]
        # Make vr and vl biorthonormal
        #vl = vl - np.dot(vr,vl)/np.dot(vr,vr)*vr
        vl /= np.dot(vl,vr)
        #print(np.sum(vr-vl))
        print('\tEnergy at site {}= {}'.format(i,E))
        M[i] = np.reshape(vr,(n1,n2,n3))
        Ml[i] = np.reshape(vl,(n1,n2,n3))
        rho[i] = np.einsum('ijk,il,ljk->',np.conj(M[i]),V_op,M[i])
        # Right Normalize
        M_reshape = np.reshape(M[i],(n1*n2,n3))
        (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
        M[i] = np.reshape(U,(n1,n2,n3))
        M[i+1] = np.einsum('i,ij,kjl->kil',s,V,M[i+1])
        M_reshape = np.reshape(Ml[i],(n1*n2,n3))
        (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
        Ml[i] = np.reshape(U,(n1,n2,n3))
        Ml[i+1] = np.einsum('i,ij,kjl->kil',s,V,Ml[i+1])
        #print(np.sum(M[i]-Ml[i]))
        # Update F
        F[i+1] = np.einsum('jlp,ijk,lmin,npq->kmq',F[i],np.conj(M[i]),W[i],M[i])
        Fl[i+1] = np.einsum('jlp,ijk,lmin,npq->kmq',Fl[i],np.conj(Ml[i]),W[i],Ml[i])
# Left Sweep -----------------------------
    print('Left Sweep {}'.format(iterCnt))
    for i in range(N-1,0,-1):
        H = np.einsum('jlp,lmin,kmq->ijknpq',F[i],W[i],F[i+1])
        (n1,n2,n3,n4,n5,n6) = H.shape
        H = np.reshape(H,(n1*n2*n3,n4*n5*n6))
        u,vl,vr = la.eig(H,left=True)
        # select max eigenvalue
        max_ind = np.argsort(u)[-1]
        E = u[max_ind]
        vr = vr[:,max_ind]
        vl = vl[:,max_ind]
        # Make vr and vl biorthonormal
        #vl = vl - np.dot(vr,vl)/np.dot(vr,vr)*vr
        vl /= np.dot(vl,vl)
        #print(np.sum(vr-vl))
        print('\tEnergy at site {}= {}'.format(i,E))
        M[i] = np.reshape(vr,(n1,n2,n3))
        Ml[i] = np.reshape(vl,(n1,n2,n3))
        rho[i] = np.einsum('ijk,il,ljk->',np.conj(M[i]),V_op,M[i])
        # Right Normalize 
        M_reshape = np.swapaxes(M[i],0,1)
        M_reshape = np.reshape(M_reshape,(n2,n1*n3))
        (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
        M_reshape = np.reshape(V,(n2,n1,n3))
        M[i] = np.swapaxes(M_reshape,0,1)
        M[i-1] = np.einsum('klj,ji,i->kli',M[i-1],U,s)
        M_reshape = np.swapaxes(Ml[i],0,1)
        M_reshape = np.reshape(M_reshape,(n2,n1*n3))
        (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
        M_reshape = np.reshape(V,(n2,n1,n3))
        Ml[i] = np.swapaxes(M_reshape,0,1)
        Ml[i-1] = np.einsum('klj,ji,i->kli',Ml[i-1],U,s)
        #print(np.sum(M[i]-Ml[i]))
        # Update F
        F[i] = np.einsum('bxc,ydbe,eaf,cdf->xya',np.conj(M[i]),W[i],M[i],F[i+1])
        Fl[i] = np.einsum('bxc,ydbe,eaf,cdf->xya',np.conj(Ml[i]),W[i],Ml[i],Fl[i+1])
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

# Resulting Wavefunction #####################
occ = np.zeros((2**N,N),dtype=int)
sum_occ = np.zeros(2**N)
for i in range(2**N):
    occ[i,:] = np.asarray(list(map(lambda x: int(x),'0'*(N-len(bin(i)[2:]))+bin(i)[2:])))
    sum_occ[i] = np.sum(occ[i,:])
# Sort Wavefunction Occupations
if False:
    inds = np.argsort(sum_occ)
    sum_occ = sum_occ[inds]
    occ = occ[inds,:]
lwf_dmrg = np.zeros(2**N,dtype=np.complex128)
rwf_dmrg = np.zeros(2**N,dtype=np.complex128)
# Calculate Wavefunction
for i in range(2**N):
    i_occ = occ[i,:]
    tmp_mat = np.array([[1]])
    tmp_matl = np.array([[1]])
    for k in range(N):
        tmp_mat = np.einsum('ij,jk->ik',tmp_mat,M[k][i_occ[k],:,:])
        tmp_matl = np.einsum('ij,jk->ik',tmp_matl,Ml[k][i_occ[k],:,:])
        #print(np.sum(tmp_mat-tmp_matl))
    rwf_dmrg[i] = tmp_mat[0,0]
    lwf_dmrg[i] = tmp_matl[0,0]
#lwf_dmrg = lwf_dmrg/np.sum(lwf_dmrg*rwf_dmrg)
##############################################



# Run Ushnish Code to Compare ################
import exactDiag_meanField
ed = exactDiag_meanField.exactDiag(L=N,
                                  clumpSize=N,
                                  alpha=alpha,
                                  gamma=0,
                                  beta=0,
                                  delta=beta,
                                  s=bias,
                                  p=1,
                                  q=0,
                                  maxIter=1000,
                                  tol=1e-10)
E_ed = ed.kernel()
print('Ushnish Exact Diagonalization Energy = {}'.format(E_ed))
print('Right Eigenvector:')
print('\tDMRG Normalization Check: {}'.format(np.sum(rwf_dmrg**2)))
print('\tED   Normalization Check: {}'.format(np.sum(rwf_ed**2)))
print('\tU_ED Normalization Check: {}'.format(np.sum(ed.rpsi**2)))
print('\tDMRG & ED    Coincidence: {}'.format(np.sum(np.abs(rwf_dmrg)-np.abs(rwf_ed))))
print('\tDMRG & U_ED  Coincidence: {}'.format(np.sum(np.abs(rwf_dmrg)-np.abs(ed.rpsi))))
print('Leftt Eigenvector:')
print('\tDMRG Normalization Check: {}'.format(np.sum(lwf_dmrg**2)))
print('\tED   Normalization Check: {}'.format(np.sum(lwf_ed**2)))
print('\tU_ED Normalization Check: {}'.format(np.sum(ed.lpsi**2)))
print('\tDMRG & ED Coincidence: {}'.format(np.sum(np.abs(lwf_dmrg)-np.abs(lwf_ed))))
print('\tDMRG & U_ED  Coincidence: {}'.format(np.sum(np.abs(lwf_dmrg)-np.abs(ed.lpsi))))
#print('LR Eigenvector:')
#print('\tDMRG Normalization Check: {}'.format(np.sum(lwf_dmrg*rwf_dmrg)))
#print('\tED   Normalization Check: {}'.format(np.sum(lwf_ed*rwf_ed)))
#print('\tU_ED Normalization Check: {}'.format(np.sum(ed.lpsi*ed.rpsi)))
np.set_printoptions(precision=1)
print('My ED (L)\tU ED (L)\t DMRG (L) \t DMRG (R)')
for i in range(len(ed.lpsi)):
    print('{}\t\t{}\t\t{}\t\t{}'.format(np.real(np.round(lwf_ed[i],3)),np.real(np.round(ed.lpsi[i],3)),np.real(np.round(lwf_dmrg[i],3)),np.real(np.round(rwf_dmrg[i],3))))
##############################################


# Calculate and compare currents #############
print('Exact Diagonalization Density')
print(np.real(ed.nv))
print('DMRG Density')
print(np.real(1-rho))
##############################################
