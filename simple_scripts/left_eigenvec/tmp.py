import numpy as np
import scipy.linalg as la
import time
np.set_printoptions(precision=10,linewidth=250)
from pyscf.lib import eig

######## Inputs ############################################################################
# SEP Model
N = 3
alpha = 0.35  # In at left
beta = 2/3    # Exit at right
s = -1         # Exponential weighting
sdiff = 0.01  
p = 1         # Jump right
# Optimization
tol = 1e-5
maxIter = 10
maxBondDim = 100
############################################################################################

######## MPO ###############################################################################
Sp = np.array([[0,1],[0,0]])
Sm = np.array([[0,0],[1,0]])
n = np.array([[0,0],[0,1]])
v = np.array([[1,0],[0,0]])
I = np.array([[1,0],[0,1]])
z = np.array([[0,0],[0,0]])
W0 = []
W1 = []
W2 = []
W0.insert(len(W0),np.array([[alpha*(np.exp(-s)*Sm-v),np.exp(-s)*Sp,-n,I]]))
W1.insert(len(W1),np.array([[alpha*(np.exp(-(s+sdiff))*Sm-v),np.exp(-(s+sdiff))*Sp,-n,I]]))
W2.insert(len(W2),np.array([[alpha*(np.exp(-(s-sdiff))*Sm-v),np.exp(-(s-sdiff))*Sp,-n,I]]))
for i in range(N-2):
    W0.insert(len(W0),np.array([[I,z,z,z],[Sm,z,z,z],[v,z,z,z],[z,np.exp(-s)*Sp,-n,I]]))
    W1.insert(len(W2),np.array([[I,z,z,z],[Sm,z,z,z],[v,z,z,z],[z,np.exp(-(s+sdiff))*Sp,-n,I]]))
    W2.insert(len(W1),np.array([[I,z,z,z],[Sm,z,z,z],[v,z,z,z],[z,np.exp(-(s-sdiff))*Sp,-n,I]]))
W0.insert(len(W0),np.array([[I],[Sm],[v],[beta*(np.exp(-s)*Sp-n)]]))
W1.insert(len(W1),np.array([[I],[Sm],[v],[beta*(np.exp(-(s+sdiff))*Sp-n)]]))
W2.insert(len(W2),np.array([[I],[Sm],[v],[beta*(np.exp(-(s-sdiff))*Sp-n)]]))
############################################################################################

# Exact Diagonalization ####################################################################
H0 = np.zeros((2**N,2**N))
H1 = np.zeros((2**N,2**N))
H2 = np.zeros((2**N,2**N))
occ = np.zeros((2**N,N),dtype=int)
sum_occ = np.zeros(2**N,dtype=int)
for i in range(2**N):
    occ[i,:] = np.asarray(list(map(lambda x: int(x),'0'*(N-len(bin(i)[2:]))+bin(i)[2:])))
    #print(occ[i,:])
    sum_occ[i] = np.sum(occ[i,:])
# Calculate Hamiltonian
for i in range(2**N):
    i_occ = occ[i,:]
    for j in range(2**N):
        j_occ = occ[j,:]
        tmp_mat0 = np.array([[1]])
        tmp_mat1 = np.array([[1]])
        tmp_mat2 = np.array([[1]])
        for k in range(N):
            tmp_mat0 = np.einsum('ij,jk->ik',tmp_mat0,W0[k][:,:,i_occ[k],j_occ[k]])
            tmp_mat1 = np.einsum('ij,jk->ik',tmp_mat1,W1[k][:,:,i_occ[k],j_occ[k]])
            tmp_mat2 = np.einsum('ij,jk->ik',tmp_mat2,W2[k][:,:,i_occ[k],j_occ[k]])
        H0[i,j] += tmp_mat0[[0]]
        H1[i,j] += tmp_mat1[[0]]
        H2[i,j] += tmp_mat2[[0]]
# Diagonalize Hamiltonian
e0,lwf_ed0,rwf_ed0 = la.eig(H0,left=True)
e1,lwf_ed1,rwf_ed1 = la.eig(H1,left=True)
e2,lwf_ed2,rwf_ed2 = la.eig(H2,left=True)
inds = np.argsort(e0)
lwf_ed0 = lwf_ed0[:,inds[-1]]
rwf_ed0 = rwf_ed0[:,inds[-1]]
inds = np.argsort(e1)
lwf_ed1 = lwf_ed1[:,inds[-1]]
rwf_ed1 = rwf_ed1[:,inds[-1]]
inds = np.argsort(e2)
lwf_ed2 = lwf_ed2[:,inds[-1]]
rwf_ed2 = rwf_ed2[:,inds[-1]]
# Ensure Proper Normalization
# <-|R> = 1
# <L|R> = 1
rwf_ed0 = rwf_ed0/np.sum(rwf_ed0)
lwf_ed0 = lwf_ed0/np.sum(lwf_ed0*rwf_ed0)
rwf_ed1 = rwf_ed1/np.sum(rwf_ed1)
lwf_ed1 = lwf_ed1/np.sum(lwf_ed1*rwf_ed1)
rwf_ed2 = rwf_ed2/np.sum(rwf_ed2)
lwf_ed2 = lwf_ed2/np.sum(lwf_ed2*rwf_ed2)
print('\nExact Diagonalization Energy: {}'.format(e0[inds[-1]]))
e0 = e0[inds[-1]]
e1 = e1[inds[-1]]
e2 = e2[inds[-1]]
print('\nExact Current: {}'.format((e1-e2)/(2*sdiff)))
#print('\nOccupation\t\t\tred\t\t\tled')
#print('-'*100)
#for i in range(len(rwf_ed)):
#    print('{}\t\t\t{},\t{}'.format(occ[i,:],rwf_ed[i],lwf_ed[i]))
############################################################################################

# Try To come up with an operator connecting left and right states
exa = alpha*np.exp(-s)
exb = beta*np.exp(-s)
exp = np.exp(-s)
if N == 2:
    #               00, 01, 10, 11 
    H = np.array([[  0,exb,  0,  0], #00
                  [  0,  0,exp,  0], #01
                  [exa,  0,  0,exb], #10
                  [  0,exa,  0,  0]])#11
    print(H)
elif N == 3:
    #              000,001,010,011,100,101,110,111
    H = np.array([[  0,exb,  0,  0,  0,  0,  0,  0], #000
                  [  0,  0,exp,  0,  0,  0,  0,  0], #001
                  [  0,  0,  0,exb,exp,  0,  0,  0], #010
                  [  0,  0,  0,  0,  0,exp,  0,  0], #011
                  [exa,  0,  0,  0,  0,exb,  0,  0], #100
                  [  0,exa,  0,  0,  0,  0,exp,  0], #101
                  [  0,  0,exa,  0,  0,  0,  0,exb], #110
                  [  0,  0,  0,exa,  0,  0,  0,  0]])#111
    print(H)
elif N == 4:
    #              0000,0001,0010,0011,0100,0101,0110,0111,1000,1001,1010,1011,1100,1101,1110,1111
    H = np.array([[   0, exb,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0], #0000
                  [   0,   0, exp,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0], #0001
                  [   0,   0,   0, exb, exp,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0], #0010
                  [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0], #0011
                  [   0,   0,   0,   0,   0, exb,   0,   0, exp,   0,   0,   0,   0,   0,   0,   0], #0100
                  [   0,   0,   0,   0,   0,   0, exp,   0,   0, exp,   0,   0,   0,   0,   0,   0], #0101
                  [   0,   0,   0,   0,   0, exp,   0, exb,   0,   0, exp,   0,   0,   0,   0,   0], #0110
                  [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, exp,   0,   0,   0,   0], #0111
                  [ exa,   0,   0,   0,   0,   0,   0,   0,   0, exb,   0,   0,   0,   0,   0,   0], #1000
                  [   0, exa,   0,   0,   0,   0,   0,   0,   0,   0, exp,   0,   0,   0,   0,   0], #1001
                  [   0,   0, exa,   0,   0,   0,   0,   0,   0,   0,   0, exb, exp,   0,   0,   0], #1010
                  [   0,   0,   0, exa,   0,   0,   0,   0,   0,   0,   0,   0,   0, exp,   0,   0], #1011
                  [   0,   0,   0,   0, exa,   0,   0,   0,   0,   0,   0,   0,   0, exb,   0,   0], #1100
                  [   0,   0,   0,   0,   0, exa,   0,   0,   0,   0,   0,   0,   0,   0, exp,   0], #1101
                  [   0,   0,   0,   0,   0,   0, exa,   0,   0,   0,   0,   0,   0,   0,   0, exb], #1110
                  [   0,   0,   0,   0,   0,   0,   0, exa,   0,   0,   0,   0,   0,   0,   0,   0]])#1111
current = np.dot(lwf_ed0,np.dot(H,rwf_ed0))
current = np.dot(np.conj(lwf_ed0).T,np.dot(H,lwf_ed0))
print('\nCurrent: {}'.format(current))
