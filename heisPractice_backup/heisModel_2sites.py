import numpy as np

# Model Parameters ##########################################
L = 2
d = 2
h = 1
J = 1
# MPO for heisenberg model
s_hat = np.array([[[0,1],  [1,0]],
                 [[0,-1j],[1j,0]],
                 [[1,0],  [0,-1]]])
zero_mat = np.zeros([2,2])
I = np.eye(2)
w_arr = np.array([[I,                   zero_mat,               zero_mat,               zero_mat,               zero_mat],
                      [s_hat[0,:,:],         zero_mat,               zero_mat,               zero_mat,               zero_mat],
                      [s_hat[1,:,:],         zero_mat,               zero_mat,               zero_mat,               zero_mat],
                      [s_hat[2,:,:],         zero_mat,               zero_mat,               zero_mat,               zero_mat],
                      [-h*s_hat[0,:,:], -J/2*s_hat[0,:,:], -J/2*s_hat[1,:,:], -J/2*s_hat[2,:,:], I       ]])
# Create initial guess ######################################
psi = np.array([[1,0],[0,0]])
(u,s,v) = np.linalg.svd(psi,full_matrices=0)
a_curr = min(d**(1),d**(L))
v = np.transpose(v)
B = [[],[]]
v = v.reshape(a_curr*d,-1)
for j in range(d):
    B[j].insert(0,v[j*a_curr:(j+1)*a_curr,:])
psi = np.dot(u,np.diag(s)).reshape(d**(L-(2)),-1)
(u,s,v) = np.linalg.svd(psi,full_matrices=0)
a_curr = min(d**(1),d**(2))
v = np.transpose(v).reshape(a_curr*d,-1)
for j in range(d):
    B[j].insert(0,v[j*a_curr:(j+1)*a_curr,:])

