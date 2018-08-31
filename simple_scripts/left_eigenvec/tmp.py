import numpy as np
import scipy.linalg as la
import time
np.set_printoptions(precision=10,linewidth=250)
from pyscf.lib import eig

######## Inputs ############################################################################
# SEP Model
N = 8
alpha = 0.35  # In at left
beta = 2/3    # Exit at right
s = -1         # Exponential weighting
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
W = []
W.insert(len(W),np.array([[alpha*(np.exp(-s)*Sm-v),np.exp(-s)*Sp,-n,I]]))
for i in range(N-2):
    W.insert(len(W),np.array([[I,z,z,z],[Sm,z,z,z],[v,z,z,z],[z,np.exp(-s)*Sp,-n,I]]))
W.insert(len(W),np.array([[I],[Sm],[v],[beta*(np.exp(-s)*Sp-n)]]))
############################################################################################

# Exact Diagonalization ####################################################################
H = np.zeros((2**N,2**N))
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
        tmp_mat = np.array([[1]])
        for k in range(N):
            tmp_mat = np.einsum('ij,jk->ik',tmp_mat,W[k][:,:,i_occ[k],j_occ[k]])
        H[i,j] += tmp_mat[[0]]
# Diagonalize Hamiltonian
e,lwf_ed,rwf_ed = la.eig(H,left=True)
inds = np.argsort(e)
lwf_ed = lwf_ed[:,inds[-1]]
rwf_ed = rwf_ed[:,inds[-1]]
# Ensure Proper Normalization
# <-|R> = 1
# <L|R> = 1
rwf_ed = rwf_ed/np.sum(rwf_ed)
lwf_ed = lwf_ed/np.sum(lwf_ed*rwf_ed)
print('\nExact Diagonalization Energy: {}'.format(e[inds[-1]]))
print('\nOccupation\t\t\tred\t\t\tled')
print('-'*100)
for i in range(len(rwf_ed)):
    print('{}\t\t\t{},\t{}'.format(occ[i,:],rwf_ed[i],lwf_ed[i]))
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
current = np.dot(lwf_ed,np.dot(H,rwf_ed))
print('\nCurrent: {}'.format(current))
