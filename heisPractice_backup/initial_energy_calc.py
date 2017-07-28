import numpy as np
n = 2
d = 2
h = 0
J = 1

# Generate right normalized MPS #################################################################################################
for i in range(n):
    if i == 0:
        if n == 2:
            psi = np.array([[-8.64312525e-14,7.07106781e-01],
                            [-7.07106781e-01,-1.14093858e-15]])
            psi = psi.reshape(d**(n-1),d)
        elif n == 4:
            psi = np.array([[ -3.69709344e-16,-5.99644631e-17,-1.69161616e-16,1.49429245e-01],
                            [ -1.34581880e-16,4.08248290e-01,-5.57677536e-01,-2.46634235e-16],
                            [ -9.68065955e-17,-5.57677536e-01,4.08248290e-01,-2.44760762e-16],
                            [  1.49429245e-01,-3.31378253e-16,-3.16743521e-16,3.18616257e-16]])
            psi = psi.reshape(d**(n-1),d)
        B = [[] for x in range(d)]
        a_prev = 1
    else:
        psi = np.dot(u,np.diag(s)).reshape(d**(n-(i+1)),-1)
        a_prev = a_curr
    (u,s,v) = np.linalg.svd(psi,full_matrices=0)
    a_curr = min(d**(i+1),d**(n-i))
    v = np.transpose(v)
    for j in range(d):
        if a_curr > a_prev:
            v = v.reshape(a_curr*d,-1)
            B[j].insert(0,v[j*a_curr:(j+1)*a_curr,:])
        else:
            v = v.reshape(-1,a_curr*d)
            B[j].insert(0,v[:,j*a_curr:(j+1)*a_curr])
M = B

# Generate MPO ###################################################################################################################
s_hat = np.array([[[0,1],  [1,0]],
                 [[0,-1j],[1j,0]], 
                 [[1,0],  [0,-1]]])
zero_mat = np.zeros([2,2])
I = np.eye(2)
# Construct MPO
w_arr = np.array([[I,               zero_mat,          zero_mat,          zero_mat,          zero_mat],
                  [s_hat[0,:,:],    zero_mat,          zero_mat,          zero_mat,          zero_mat],
                  [s_hat[1,:,:],    zero_mat,          zero_mat,          zero_mat,          zero_mat],
                  [s_hat[2,:,:],    zero_mat,          zero_mat,          zero_mat,          zero_mat],
                  [-h*s_hat[0,:,:], -J/2*s_hat[0,:,:], -J/2*s_hat[1,:,:], -J/2*s_hat[2,:,:], I       ]])

def W(ind,w_arr,n):
    if ind == 0:
        return np.expand_dims(w_arr[-1,:],0)
    elif ind == n-1:
        return np.expand_dims(w_arr[:,0],1)
    else:
        return w_arr

# Calculate R-Expressions ########################################################################################################
R = []
R.insert(0,np.array([[[1]]]))
for out_cnt in range(n)[::-1]:
    for i in range(d):
        for j in range(d):
            if out_cnt == 0:
                tmp_array = np.array([[[1]]])
            else:
                if i+j == 0:
                    tmp_array = np.einsum('ij,kl,mn,jln->ikm',np.conjugate(M[i][out_cnt]),W(out_cnt,w_arr,n)[:,:,i,j],M[j][out_cnt],R[0])
                else:
                    tmp_array += np.einsum('ij,kl,mn,jln->ikm',np.conjugate(M[i][out_cnt]),W(out_cnt,w_arr,n)[:,:,i,j],M[j][out_cnt],R[0])
    R.insert(0,tmp_array)

# Calculate Energy ###############################################################################################################
site = 0
for i in range(d):
    for j in range(d):
        if i+j == 0:
            numerator = np.einsum('ijk,jl,olp,io,kp->',np.array([[[1]]]),W(site,w_arr,n)[:,:,i,j],R[site+1],np.conjugate(M[i][site]),M[i][site])
        else: 
            numerator = np.einsum('ijk,jl,olp,io,kp->',np.array([[[1]]]),W(site,w_arr,n)[:,:,i,j],R[site+1],np.conjugate(M[i][site]),M[i][site])
print('Calculated Energy: {}'.format(numerator))
