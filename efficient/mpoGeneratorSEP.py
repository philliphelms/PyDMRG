import numpy as np

N = 12
ham_dim = 10+(N-2)*4
W = np.zeros((ham_dim,ham_dim))

# Construct left column
W[0,0] = 1
W[1,0] = 2 # ju*Sm
W[N,0] = 3 # jr*Sm
W[N+1,0] = 4 # ju*v
W[2*N,0] = 5 # jr*v
W[2*N+1,0] = 6 # jd*Sp
W[3*N,0] = 7 # jl*Sp
W[3*N+1,0] = 8 # jd*n
W[4*N,0] = 9 # jl*n
W[4*N+1,0] = 10 # inout

# Construct interior of MPO
col_ind = 1
row_ind = 2
for j in range(4):
    for i in range(N-1):
        W[row_ind,col_ind] = 1
        row_ind += 1
        col_ind += 1
    row_ind += 1
    col_ind += 1

# Construct bottom row
W[-1,0] = 10 # inout
W[-1,N] = 11 # exp*Sp
W[-1,2*N] = 12 # n
W[-1,3*N] = 13 # exp*Sm
W[-1,4*N] = 14 # -v
W[-1,4*N+1] = 1

print(W)
