import numpy as np
import scipy.linalg as la
np.set_printoptions(precision=3,linewidth=1000)

######## Inputs ############################################################################
# SEP Model
N = 8
alpha = 0.35  # In at left
beta = 2/3    # Exit at right
s = -1        # Exponential weighting
p = 1         # Jump right
# Optimization
tol = 1e-5
maxIter = 10
maxBondDim = 16
############################################################################################

######## MPO ###############################################################################
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
############################################################################################

# Exact Diagonalization ####################################################################
H = np.zeros((2**N,2**N))
occ = np.zeros((2**N,N),dtype=int)
sum_occ = np.zeros(2**N)
for i in range(2**N):
    occ[i,:] = np.asarray(list(map(lambda x: int(x),'0'*(N-len(bin(i)[2:]))+bin(i)[2:])))
    sum_occ[i] = np.sum(occ[i,:])
# Calculate Hamiltonian
for i in range(2**N):
    i_occ = occ[i,:]
    for j in range(2**N):
        j_occ = occ[j,:]
        tmp_mat = np.array([[1]])
        for k in range(N):
            tmp_mat = np.einsum('ij,jk->ik',tmp_mat,W[k][:,:,i_occ[k],j_occ[k]])
        H[i,j] += tmp_mat[[0]]
# Diagonalize Hamiltonian
e,lwf_ed,rwf_ed = la.eig(H,left=True)
inds = np.argsort(e)
lwf_ed =lwf_ed[:,inds[-1]]
rwf_ed = rwf_ed[:,inds[-1]]
# Ensure Proper Normalization
# <-|R> = 1
# <L|R> = 1
rwf_ed = rwf_ed/np.sum(rwf_ed)
lwf_ed = lwf_ed/np.sum(lwf_ed*rwf_ed)
print(rwf_ed)
print(np.sum(rwf_ed))
print(lwf_ed)
############################################################################################

# Decompose States into MPS ################################################################
psir = np.zeros([2]*N,dtype=np.complex128)
psil = np.zeros([2]*N,dtype=np.complex128)
occ = np.zeros((2**N,N),dtype=int)
for i in range(2**N):
    occ[i,:] = np.asarray(list(map(lambda x: int(x),'0'*(N-len(bin(i)[2:]))+bin(i)[2:])))
for i in range(2**N):
    psir[tuple(occ[i,:])] = rwf_ed[i]
    psil[tuple(occ[i,:])] = lwf_ed[i]
# Determine Matrix Dimensions
fbd_site = []
mbd_site = []
fbd_site.insert(0,1)
mbd_site.insert(0,1)
for i in range(int(N/2)):
    fbd_site.insert(-1,2**i)
    mbd_site.insert(-1,min(2**i,maxBondDim))
for i in range(int(N/2))[::-1]:
    fbd_site.insert(-1,2**(i+1))
    mbd_site.insert(-1,min(2**(i+1),maxBondDim))
# Decompose Wavefunction from the right
Mr = [] # Ordering (sigma,a_0,a_1)
Ml = []
for i in range(N,1,-1):
    psir = np.reshape(psir,(2**(i-1),-1))
    psil = np.reshape(psil,(2**(i-1),-1))
    (ur,sr,vr) = np.linalg.svd(psir,full_matrices=False)
    (ul,sl,vl) = np.linalg.svd(psil,full_matrices=False)
    # make left eigenvector right-canonical
    Xgauge = np.dot(vr,np.linalg.pinv(vl))
    vl = np.dot(Xgauge,vl)
    Br = np.reshape(vr,(fbd_site[i-1],2,mbd_site[i]))
    Bl = np.reshape(vl,(fbd_site[i-1],2,mbd_site[i]))
    Br = Br[:mbd_site[i-1],:,:mbd_site[i]] 
    Bl = Bl[:mbd_site[i-1],:,:mbd_site[i]]
    Br = np.swapaxes(Br,0,1)
    Bl = np.swapaxes(Bl,0,1)
    Mr.insert(0,Br)
    Ml.insert(0,Bl)
    psir = np.einsum('ij,j->ij',ur[:,:mbd_site[i-1]],sr)
    psil = np.einsum('ij,j,jk->ik',ul[:,:mbd_site[i-1]],sl,np.linalg.pinv(Xgauge))
    #psil = np.einsum('ij,j->ij',ul[:,:mbd_site[i-1]],sl)
    # Check for Correct Canonicalization?
    print('Check for Both  normalization:\n{}'.format(np.einsum('ijk,ikl->jl',Ml[0],np.transpose(Mr[0],(0,2,1)))))
Mr.insert(0,np.reshape(psir,(2,1,min(2,maxBondDim))))
Ml.insert(0,np.reshape(psil,(2,1,min(2,maxBondDim))))
##############################################

# Now Calculate State from MPS ###############
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
    tmp_matr = np.array([[1]])
    tmp_matl = np.array([[1]])
    for k in range(N):
        tmp_matr = np.einsum('ij,jk->ik',tmp_matr,Mr[k][i_occ[k],:,:])
        tmp_matl = np.einsum('ij,jk->ik',tmp_matl,Ml[k][i_occ[k],:,:])
    #print(np.sum(tmp_mat-tmp_matl))
    rwf_dmrg[i] = tmp_matr[0,0]
    lwf_dmrg[i] = tmp_matl[0,0]
print(rwf_dmrg)
print(lwf_dmrg)
##############################################

"""
# Create F ###################################
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
# Calculate Initial Values
for i in range(int(N)-1,0,-1):
    F[i] = np.einsum('bxc,ydbe,eaf,cdf->xya',np.conj(M[i]),W[i],M[i],F[i+1])
for i in range(int(N)-1,0,-1):
    Fl[i] = np.einsum('bxc,ydbe,eaf,cdf->xya',np.conj(Ml[i]),W[i],Ml[i],Fl[i+1])
##############################################

# Optimization Sweeps ########################
converged = False
iterCnt = 0
E_prev = 0
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
        print('\tEnergy at site {}= {}'.format(i,E))
        print('Before Opt (R): {}'.format(np.real(np.reshape(M[i],-1))))
        M[i] = np.reshape(vr,(n1,n2,n3))
        print('After  Opt (R): {}'.format(np.real(np.reshape(M[i],-1))))
        print('Before Opt (L): {}'.format(np.real(np.reshape(Ml[i],-1))))
        Ml[i] = np.reshape(vl,(n1,n2,n3))
        print('After  Opt (L): {}'.format(np.real(np.reshape(Ml[i],-1))))
        # Right Normalize
        M_reshape = np.reshape(M[i],(n1*n2,n3))
        (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
        M[i] = np.reshape(U,(n1,n2,n3))
        M[i+1] = np.einsum('i,ij,kjl->kil',s,V,M[i+1])
        M_reshape = np.reshape(Ml[i],(n1*n2,n3))
        (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
        Ml[i] = np.reshape(U,(n1,n2,n3))
        Ml[i+1] = np.einsum('i,ij,kjl->kil',s,V,M[i+1])
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
        print(vl)
        #print('vr = {}'.format(vr))
        #print('vl = {}'.format(vl))
        print('\tEnergy at site {}= {}'.format(i,E))
        M[i] = np.reshape(vr,(n1,n2,n3))
        Ml[i] = np.reshape(vl,(n1,n2,n3))
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
        Ml[i-1] = np.einsum('klj,ji,i->kli',M[i-1],U,s)
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

# Now Calculate State from MPS ###############
occ = np.zeros((2**N,N),dtype=int)
sum_occ = np.zeros(2**N)
for i in range(2**N):
    occ[i,:] = np.asarray(list(map(lambda x: int(x),'0'*(N-len(bin(i)[2:]))+bin(i)[2:])))
    sum_occ[i] = np.sum(occ[i,:])
# Calculate Wavefunction
rwf_dmrg = np.zeros(2**N,dtype=np.complex128)
for i in range(2**N):
    i_occ = occ[i,:]
    tmp_mat = np.array([[1]])
    for k in range(N):
        tmp_mat = np.einsum('ij,jk->ik',tmp_mat,M[k][i_occ[k],:,:])
    #print(np.sum(tmp_mat-tmp_matl))
    rwf_dmrg[i] = tmp_mat[0,0]
##############################################
"""
