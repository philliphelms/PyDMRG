#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# Adapted for pydmrg

import ctypes
import numpy as np
import re

'''
Extensions to numpy module
'''

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)


def load_library(libname):
    # numpy 1.6 has bug in ctypeslib.load_library, see numpy/distutils/misc_util.py
    if '1.6' in np.__version__:
        if (sys.platform.startswith('linux') or
            sys.platform.startswith('gnukfreebsd')):
            so_ext = '.so'
        elif sys.platform.startswith('darwin'):
            so_ext = '.dylib'
        elif sys.platform.startswith('win'):
            so_ext = '.dll'
        else:
            raise OSError('Unknown platform')
        libname_so = libname + so_ext
        return ctypes.CDLL(os.path.join(os.path.dirname(__file__), libname_so))
    else:
        _loaderpath = os.path.dirname(__file__)
        return np.ctypeslib.load_library(libname, _loaderpath)

_np_helper = load_library('libnp_helper')

def transpose(a, axes=None, inplace=False, out=None):
    '''Transpose array for better memory efficiency

    Examples:

    >>> transpose(np.ones((3,2)))
    [[ 1.  1.  1.]
     [ 1.  1.  1.]]
    '''
    if inplace:
        arow, acol = a.shape
        assert(arow == acol)
        tmp = np.empty((BLOCK_DIM,BLOCK_DIM))
        for c0, c1 in prange(0, acol, BLOCK_DIM):
            for r0, r1 in prange(0, c0, BLOCK_DIM):
                tmp[:c1-c0,:r1-r0] = a[c0:c1,r0:r1]
                a[c0:c1,r0:r1] = a[r0:r1,c0:c1].T
                a[r0:r1,c0:c1] = tmp[:c1-c0,:r1-r0].T
            # diagonal blocks
            a[c0:c1,c0:c1] = a[c0:c1,c0:c1].T
        return a

    if not a.flags.c_contiguous:
        if a.ndim == 2:
            arow, acol = a.shape
            out = np.empty((acol,arow), a.dtype)
            r1 = c1 = 0
            for c0 in range(0, acol-BLOCK_DIM, BLOCK_DIM):
                c1 = c0 + BLOCK_DIM
                for r0 in range(0, arow-BLOCK_DIM, BLOCK_DIM):
                    r1 = r0 + BLOCK_DIM
                    out[c0:c1,r0:r1] = a[r0:r1,c0:c1].T
                out[c0:c1,r1:arow] = a[r1:arow,c0:c1].T
            for r0 in range(0, arow-BLOCK_DIM, BLOCK_DIM):
                r1 = r0 + BLOCK_DIM
                out[c1:acol,r0:r1] = a[r0:r1,c1:acol].T
            out[c1:acol,r1:arow] = a[r1:arow,c1:acol].T
            return out
        else:
            return a.transpose(axes)

    if a.ndim == 2:
        arow, acol = a.shape
        c_shape = (ctypes.c_int*3)(1, arow, acol)
        out = np.ndarray((acol, arow), a.dtype, buffer=out)
    elif a.ndim == 3 and axes == (0,2,1):
        d0, arow, acol = a.shape
        c_shape = (ctypes.c_int*3)(d0, arow, acol)
        out = np.ndarray((d0, acol, arow), a.dtype, buffer=out)
    else:
        raise NotImplementedError

    assert(a.flags.c_contiguous)
    if a.dtype == np.double:
        fn = _np_helper.NPdtranspose_021
    else:
        fn = _np_helper.NPztranspose_021
    fn.restype = ctypes.c_void_p
    fn(c_shape, a.ctypes.data_as(ctypes.c_void_p),
       out.ctypes.data_as(ctypes.c_void_p))
    return out

# NOTE: NOT assume array a, b to be C-contiguous, since a and b are two
# pointers we want to pass in.
# np.dot might not call optimized blas
def ddot(a, b, alpha=1, c=None, beta=0):
    '''Matrix-matrix multiplication for double precision arrays
    '''
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]
    if a.flags.c_contiguous:
        trans_a = 'N'
    elif a.flags.f_contiguous:
        trans_a = 'T'
        a = a.T
    else:
        a = np.asarray(a, order='C')
        trans_a = 'N'
        #raise ValueError('a.flags: %s' % str(a.flags))

    assert(k == b.shape[0])
    if b.flags.c_contiguous:
        trans_b = 'N'
    elif b.flags.f_contiguous:
        trans_b = 'T'
        b = b.T
    else:
        b = np.asarray(b, order='C')
        trans_b = 'N'
        #raise ValueError('b.flags: %s' % str(b.flags))

    if c is None:
        c = np.empty((m,n))
        beta = 0
    else:
        assert(c.shape == (m,n))

    return _dgemm(trans_a, trans_b, m, n, k, a, b, c, alpha, beta)

def zdot(a, b, alpha=1, c=None, beta=0):
    '''Matrix-matrix multiplication for double complex arrays using Gauss's
    complex multiplication algorithm
    '''
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]
    if a.flags.c_contiguous:
        trans_a = 'N'
    elif a.flags.f_contiguous:
        trans_a = 'T'
        a = a.T
    else:
        raise ValueError('a.flags: %s' % str(a.flags))

    assert(k == b.shape[0])
    if b.flags.c_contiguous:
        trans_b = 'N'
    elif b.flags.f_contiguous:
        trans_b = 'T'
        b = b.T
    else:
        raise ValueError('b.flags: %s' % str(b.flags))

    if c is None:
        beta = 0
        c = np.empty((m,n), dtype=np.complex128)
    else:
        assert(c.shape == (m,n))

    return _zgemm(trans_a, trans_b, m, n, k, a, b, c, alpha, beta)

def dot(a, b, alpha=1, c=None, beta=0):
    atype = a.dtype
    btype = b.dtype

    if atype == np.float64 and btype == np.float64:
        if c is None or c.dtype == np.float64:
            return ddot(a, b, alpha, c, beta)
        else:
            cr = np.asarray(c.real, order='C')
            c.real = ddot(a, b, alpha, cr, beta)
            return c

    if atype == np.float64 and btype == np.complex128:
        br = np.asarray(b.real, order='C')
        bi = np.asarray(b.imag, order='C')
        cr = ddot(a, br, alpha)
        ci = ddot(a, bi, alpha)
        ab = cr + ci*1j

    elif atype == np.complex128 and btype == np.float64:
        ar = np.asarray(a.real, order='C')
        ai = np.asarray(a.imag, order='C')
        cr = ddot(ar, b, alpha)
        ci = ddot(ai, b, alpha)
        ab = cr + ci*1j

    elif atype == np.complex128 and btype == np.complex128:
        #k1 = ddot(a.real+a.imag, b.real.copy(), alpha)
        #k2 = ddot(a.real.copy(), b.imag-b.real, alpha)
        #k3 = ddot(a.imag.copy(), b.real+b.imag, alpha)
        #ab = k1-k3 + (k1+k2)*1j
        return zdot(a, b, alpha, c, beta)

    else:
        ab = np.dot(a, b) * alpha

    if c is None:
        c = ab
    else:
        if beta == 0:
            c[:] = 0
        else:
            c *= beta
        c += ab
    return c

def einsum(idx_str, *tensors):
    '''Perform a more efficient einsum via reshaping to a matrix multiply.

    Current differences compared to np.einsum:
    This assumes that each repeated index is actually summed (i.e. no 'i,i->i')
    and appears only twice (i.e. no 'ij,ik,il->jkl'). The output indices must
    be explicitly specified (i.e. 'ij,j->i' and not 'ij,j').
    '''

    DEBUG = False

    idx_str = idx_str.replace(' ','')
    indices  = "".join(re.split(',|->',idx_str))
    if '->' not in idx_str or any(indices.count(x)>2 for x in set(indices)):
        return np.einsum(idx_str,*tensors)

    if idx_str.count(',') > 1:
        indices  = re.split(',|->',idx_str)
        indices_in = indices[:-1]
        idx_final = indices[-1]
        n_shared_max = 0
        for i in range(len(indices_in)):
            for j in range(i):
                tmp = list(set(indices_in[i]).intersection(indices_in[j]))
                n_shared_indices = len(tmp)
                if n_shared_indices > n_shared_max:
                    n_shared_max = n_shared_indices
                    shared_indices = tmp
                    [a,b] = [i,j]
        tensors = list(tensors)
        A, B = tensors[a], tensors[b]
        idxA, idxB = indices[a], indices[b]
        idx_out = list(idxA+idxB)
        idx_out = "".join([x for x in idx_out if x not in shared_indices])
        C = einsum(idxA+","+idxB+"->"+idx_out, A, B)
        indices_in.pop(a)
        indices_in.pop(b)
        indices_in.append(idx_out)
        tensors.pop(a)
        tensors.pop(b)
        tensors.append(C)
        return einsum(",".join(indices_in)+"->"+idx_final,*tensors)

    A, B = tensors
    # A or B might be HDF5 Datasets
    A = np.array(A, copy=False)
    B = np.array(B, copy=False)
    if A.size < 3000 or B.size < 3000:
        return np.einsum(idx_str, *tensors)

    # Split the strings into a list of idx char's
    idxA, idxBC = idx_str.split(',')
    idxB, idxC = idxBC.split('->')
    idxA, idxB, idxC = [list(x) for x in [idxA,idxB,idxC]]

    # Get the range for each index and put it in a dictionary
    rangeA = dict()
    rangeB = dict()
    #rangeC = dict()
    for idx,rnge in zip(idxA,A.shape):
        rangeA[idx] = rnge
    for idx,rnge in zip(idxB,B.shape):
        rangeB[idx] = rnge
    #for idx,rnge in zip(idxC,C.shape):
    #    rangeC[idx] = rnge

    # Find the shared indices being summed over
    shared_idxAB = list(set(idxA).intersection(idxB))
    #if len(shared_idxAB) == 0:
    #    return np.einsum(idx_str,A,B)
    idxAt = list(idxA)
    idxBt = list(idxB)
    inner_shape = 1
    insert_B_loc = 0
    for n in shared_idxAB:
        if rangeA[n] != rangeB[n]:
            print("ERROR: In index string", idx_str, ", the range of index", n, "is different in A (%d) and B (%d)"%(
                    rangeA[n], rangeB[n]))
            raise SystemExit

        # Bring idx all the way to the right for A
        # and to the left (but preserve order) for B
        idxA_n = idxAt.index(n)
        idxAt.insert(len(idxAt)-1, idxAt.pop(idxA_n))

        idxB_n = idxBt.index(n)
        idxBt.insert(insert_B_loc, idxBt.pop(idxB_n))
        insert_B_loc += 1

        inner_shape *= rangeA[n]

    # Transpose the tensors into the proper order and reshape into matrices
    new_orderA = list()
    for idx in idxAt:
        new_orderA.append(idxA.index(idx))
    new_orderB = list()
    for idx in idxBt:
        new_orderB.append(idxB.index(idx))

    At = A.transpose(new_orderA).reshape(-1,inner_shape)
    Bt = B.transpose(new_orderB).reshape(inner_shape,-1)

    shapeCt = list()
    idxCt = list()
    for idx in idxAt:
        if idx in shared_idxAB:
            break
        shapeCt.append(rangeA[idx])
        idxCt.append(idx)
    for idx in idxBt:
        if idx in shared_idxAB:
            continue
        shapeCt.append(rangeB[idx])
        idxCt.append(idx)

    new_orderCt = list()
    for idx in idxC:
        new_orderCt.append(idxCt.index(idx))

    return np.dot(At,Bt).reshape(shapeCt).transpose(new_orderCt)
