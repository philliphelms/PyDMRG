import numpy as np


def arnoldi(aop, v0, precond, k=1):
    """
    Arnoldi algorithm (Krylov approximation of a matrix)
        input: 
            A: matrix to approximate
            v0: initial vector (should be in matrix form) 
            k: number of Krylov steps 
        output: 
            V: matrix (large, N*k) containing the orthogonal vectors
            H: matrix (small, k*k) containing the Krylov approximation of A
    Author: Vasile Gradinaru, 14.12.2007 (Rennes)
    """
    V = v0 / np.linalg.norm(v0)
    H = np.zeros((k+1,k)
    for m in xrange(k):
        vt = aop(v0)
        for j in xrange( m+1):
            H[ j, m] = (V[ :, j].H * vt )[0,0]
            vt -= H[ j, m] * V[:, j]
        H[ m+1, m] = norm(vt);
        if m is not k-1:
            V =  hstack( (V, vt.copy() / H[ m+1, m] ) ) 
    return V,  H


def arnoldi(A, v0, k):
    """
    Arnoldi algorithm (Krylov approximation of a matrix)
        input: 
            A: matrix to approximate
            v0: initial vector (should be in matrix form) 
            k: number of Krylov steps 
        output: 
            V: matrix (large, N*k) containing the orthogonal vectors
            H: matrix (small, k*k) containing the Krylov approximation of A
    Author: Vasile Gradinaru, 14.12.2007 (Rennes)
    """

    V = v0 / np.linalg.norm(v0)
    H = mat( zeros((k+1,k) )
    for m in xrange(k):
        vt = A*V[ :, m]
        for j in xrange( m+1):
            H[ j, m] = (V[ :, j].H * vt )[0,0]
            vt -= H[ j, m] * V[:, j]
        H[ m+1, m] = norm(vt);
        if m is not k-1:
            V =  hstack( (V, vt.copy() / H[ m+1, m] ) ) 
    return V,  H







def my_arnoldi(aop,v0,m=10):
    v = v0/np.linalg.norm(v0)
    h = np.zeros((k+1,k))
    for j in range(m):
        Av = aop(v)
        for i in range(j):
            h[i,j] = Av[j]*v[i]
        w[j] = Av[j]-np.einsum('ij,i->j',h,v)
        h[j+1,j] = np.linalg.norm(w)
        if np.abs(h[j+1,j]) < 1e-16:
            return 0
        v[j] = w[j]/h[j+1,j]








def arnoldi_fast(aop, v0, precond, k=1):
    """
    Arnoldi algorithm (Krylov approximation of a matrix)
        input: 
            A: matrix to approximate
            v0: initial vector (should be in matrix form) 
            k: number of Krylov steps 
        output: 
            V: matrix (large, N*k) containing the orthogonal vectors
            H: matrix (small, k*k) containing the Krylov approximation of A
    Author: Vasile Gradinaru, 14.12.2007 (Rennes)
    """
#    print 'ARNOLDI METHOD'
    V = np.zeros((v0.shape[0],k+1))
    V[:,0] = v0/np.linalg.norm(v0)
    H = np.zeros((k+1,k)
    for i in xrange(k):
        vt = A*V[ :, i]
        for j in xrange( m+1):
            H[ j, i] = (V[ :, j].H * vt )[0,0]
            vt -= H[ j, i] * V[:, j]
        H[ i+1, i] = norm(vt);
        V[:,i+1] = vt.copy() / H[ i+1, i]
    return V,  H


def arnoldi_fast_nocopy(A, v0, k):
    """
    Arnoldi algorithm (Krylov approximation of a matrix)
    Uses in-place computations and row-major format
        input:
            A: matrix to approximate
            v0: initial vector (should be in matrix form)
            k: number of Krylov steps
        output:
            V: matrix (large, N*k) containing the orthogonal vectors
            H: matrix (small, k*k) containing the Krylov approximation of A
    Author: Vasile Gradinaru, 14.12.2007 (Rennes)
    """
#    print 'ARNOLDI METHOD'
    inputtype = A.dtype.type
    n = v0.shape[0]
    V = zeros((k+1,n), dtype=inputtype)
    V[0,:] = v0.T.copy()/norm(v0)
    H = zeros((k+1,k), dtype=inputtype)
    for m in xrange(k):
        V[m+1,:] = dot(A,V[m,:])
        for j in xrange( m+1):
            H[ j, m] = dot(V[j,:], V[m+1,:])
            V[m+1,:] -= H[ j, m] * V[j,:]
        H[ m+1, m] = norm(V[m+1,:]);
        V[m+1,:] /= H[ m+1, m]
    return V.T,  H
