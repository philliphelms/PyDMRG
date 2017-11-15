import numpy as np

N = 12
ham_dim = 8+(N-2)*3
W = np.zeros((ham_dim,ham_dim))

# Construct left column
W[0,0] = 1 
W[1,0] = 2
W[N,0] = 2
W[N+1,0] = 3
W[2*N,0] = 3
W[2*N+1,0] = 4
W[3*N,0] = 4

# Construct interior of MPO
col_ind = 1
row_ind = 2
for i in range(N-1):
    W[row_ind,col_ind] = 1
    row_ind += 1
    col_ind += 1
row_ind += 1
col_ind += 1
for i in range(N-1):
    W[row_ind,col_ind] = 1
    row_ind += 1
    col_ind += 1
row_ind += 1
col_ind += 1
for i in range(N-1):
    W[row_ind,col_ind] = 1
    row_ind += 1
    col_ind += 1

# Construct bottom row
W[-1,0] = 5
W[-1,N] = 6
W[-1,2*N] = 7
W[-1,3*N] = 8
W[-1,3*N+1] = 1

print(W)
