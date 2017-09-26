import numpy as np

# Number of Sites
L = 4
# SEP Parameters
alpha = 0.5 # Enter at left
beta = 0.5  # Enter at right
delta = 0   # Leave at right
gamma = 0   # Leave at left
p = 0.5     # Hop right
q = 0       # Hop left
s = 0

# Other useful vars
s_p = np.array([[0,1],
                [0,0]])
s_m = np.array([[0,0],
                [1,0]])
n = np.array([[0,0],
              [0,1]])
v = np.array([[1,0],
              [0,0]])

# Function for calculating occupation from 
def int2stringVec(i,L):
    occString = "{0:b}".format(i)
    if len(occString) < L:
        occString = "0"*(L-len(occString)) + occString
    occVec = np.zeros([L,2])
    for j in range(len(occString)):
        occVec[j,0] = int(occString[j])
        occVec[j,1] = not int(occString[j])
    return occVec

H = np.zeros([2**L,2**L])
for i in range(L):
    occVec_i = int2stringVec(i,L)
    print(occVec_i)
    for j in range(L):
        occVec_j = int2stringVec(j,L)
        # Evaluate H|psi_j>
        H_psi = np.zeros([2])
        # Left Entry & Exit
        H_psi = alpha*(np.exp(-s)*np.dot(s_m,occVec_j[0,:])-np.dot(v,occVec_j[0,:]))+\
                gamma*(np.exp(-s)*np.dot(s_p,occVec_j[0,:])-np.dot(n,occVec_j[0,:]))
        # Right Entry & Exit
        H_psi += beta*(np.exp(-s)*np.dot(s_m,occVec_j[-1,:])-np.dot(v,occVec_j[-1,:]))+\
                delta*(np.exp(-s)*np.dot(s_p,occVec_j[-1,:])-np.dot(n,occVec_j[-1,:]))
        # Center Sites
        for k in range(L-1):
            H_psi += p*(np.exp(-s)*np.dot(np.dot(s_p,occVec_j[k,:]),np.dot(s_m,occVec_j[k+1,:]))+\
                        np.dot(np.dot(n,occVec_j[k,:]),np.dot(v,occVec_j[k+1,:])))+\
                     q*(np.exp(-s)*np.dot(np.dot(s_m,occVec_j[k,:]),np.dot(s_p,occVec_j[k+1,:]))+\
                        np.dot(np.dot(v,occVec_j[k,:]),np.dot(n,occVec_j[k+1,:])))
        print(H_psi.shape)
