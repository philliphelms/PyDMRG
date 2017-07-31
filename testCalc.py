import numpy as np

h = 0
J = 1

M = [[],[]]
M[0] = np.array([[[ -2.22e-16,   7.07e-01]],[[  7.07e-01,   1.57e-16]]])
M[1] = np.array([[[1,0]],[[0,-1]]])

psi_A = np.array([[1]])
psi_B = np.eye(2)
psi_psi = np.einsum('ij,kil,kjm,lm->',psi_A,M[0],M[1],psi_B)
print(psi_psi)

S_p = np.array([[0,1],
                [0,0]])
S_m = np.array([[0,0],
                [1,0]])
S_z = np.array([[1,0],
                [0,-1]])
zero_mat = np.zeros([2,2])
I = np.eye(2)
# Construct MPO
w_arr      = np.array([[I,           zero_mat,      zero_mat,      zero_mat,   zero_mat],
                       [S_p,         zero_mat,      zero_mat,      zero_mat,   zero_mat],
                       [S_m,         zero_mat,      zero_mat,      zero_mat,   zero_mat],
                       [S_z,         zero_mat,      zero_mat,      zero_mat,   zero_mat],
                       [-h*S_z,      J/2.*S_m,      J/2.*S_p,      J*S_z,      I       ]])

