import numpy as np
import scipy.linalg as sla
from pyscf.lib import einsum
from pyscf.lib import eig as davidson
from scipy.sparse.linalg import eigs as arnoldi
from scipy.sparse.linalg import LinearOperator
from tools.mps_tools import *
import warnings
import copy

VERBOSE = 10

def calc_diag(M,W,F,site):
    (n1,n2,n3) = M[site].shape
    diag = np.zeros((n1*n2*n3),dtype=np.complex_)
    for mpoInd in range(len(W)):
        if W[mpoInd][site] is None:
            mpo_diag = einsum('abnn->anb',np.array([[np.eye(2)]]))
        else:
            mpo_diag = einsum('abnn->anb',W[mpoInd][site])
        l_diag = einsum('lal->la',F[mpoInd][site])
        r_diag = einsum('rbr->rb',F[mpoInd][site+1])
        tmp = einsum('la,anb->lnb',l_diag,mpo_diag)
        diag_mis = einsum('lnb,rb->nlr',tmp,r_diag)
        diag += diag_mis.ravel()
    return diag

def calc_ham(M,W,F,site):
    (n1,n2,n3) = M[site].shape
    dim = n1*n2*n3
    H = np.zeros((dim,dim),dtype=np.complex_)
    for mpoInd in range(len(W)):
        if W[mpoInd][site] is None:
            tmp1 = einsum('lmin,kmq->linkq',np.array([[np.eye(2)]]),F[mpoInd][site+1])
        else:
            tmp1 = einsum('lmin,kmq->linkq',W[mpoInd][site],F[mpoInd][site+1])
        Htmp = einsum('jlp,linkq->ijknpq',F[mpoInd][site],tmp1)
        H += np.reshape(Htmp,(dim,dim))
    return H

def check_overlap(Mprev,vecs,E,preserveState=False,printStates=False,allowSwap=True,reuseOldState=True):
    vecShape = vecs.shape
    if len(vecShape) == 1: 
        nVecs = 1
        vecs = np.swapaxes(np.array([vecs]),0,1)
    else: _,nVecs = vecShape
    # Check to make sure we dont have a small gap
    matchedState = False
    for j in range(nVecs):
        ovlp_j = np.abs(np.dot(Mprev,np.conj(vecs[:,j])))
        if VERBOSE > 3: print('\t\tChecking Overlap {} = {}'.format(j,ovlp_j))
        if ovlp_j > 0.98:
            matchedState = True
            if (j != 0) and preserveState and allowSwap:
                # Swap eigenstates
                if VERBOSE > 3: print('!!! Swapping States {} & {} !!!'.format(0,j))
                tmpVec = copy.deepcopy(vecs[:,j])
                vecs[:,j] = copy.deepcopy(vecs[:,0])
                vecs[:,0] = tmpVec
                Etmp = E[j]
                E[j] = E[0]
                E[0] = Etmp
    if reuseOldState and preserveState and (not matchedState):
        if VERBOSE > 3: print('!!! Correct State Not Found !!!')
        vecs = Mprev
        # Reformat it so it matches vecs
        vecs = np.swapaxes(np.array([vecs]),0,1)
    if printStates:
        if np.abs(np.dot(Mprev,np.conj(vecs[:,0]))) < 0.9:
            print('Guess Prev\t\tEig Vec 0\t\tEig Vec 1')
            indices = np.argsort(np.abs(Mprev))[::-1]
            for i in range(len(Mprev)):
                print('{}\t{}\t{}'.format(Mprev[indices[i]],vecs[indices[i],0],vecs[indices[i],1]))
    return E,vecs,np.abs(np.dot(Mprev,np.conj(vecs[:,0])))

def make_ham_func_oneSite(M,W,F,site,usePrecond=False,debug=False):
    # Define Hamiltonian function to give Hx
    def Hfun(x):
        x_reshape = np.reshape(x,M[site].shape)
        fin_sum = np.zeros(x_reshape.shape,dtype=np.complex_)
        # Loop over all MPOs
        for mpoInd in range(len(W)):
            if W[mpoInd][site] is None:
                in_sum1 = einsum('ijk,lmk->ijlm',F[mpoInd][site+1],x_reshape)
                fin_sum +=einsum('pnm,inom->opi',F[mpoInd][site],in_sum1)
            else:
                in_sum1 = einsum('ijk,lmk->ijlm',F[mpoInd][site+1],x_reshape)
                in_sum2 = einsum('njol,ijlm->noim',W[mpoInd][site],in_sum1)
                fin_sum +=einsum('pnm,noim->opi',F[mpoInd][site],in_sum2)
        # If desired, compare Hx function to analytic Hx
        if debug:
            H = calc_ham(M,W,F,site)
            assert(np.isclose(np.sum(np.abs(np.reshape(fin_sum,-1)-np.dot(H,x))),0))
        # Return flattened result
        return -np.reshape(fin_sum,-1)
    # Precond often slows convergence
    if usePrecond:
        diag = calc_diag(M,W,F,site)
        # Compare analytic diagonal and calculated
        if debug:
            H = calc_ham(M,W,F,site)
            assert(np.isclose(np.sum(np.abs(diag-np.diag(H))),0))
        # Return Davidson Preconditioner
        def precond(dx,e,x0):
            return dx/diag-e
    else:
        # Return original values, often speeds convergence
        def precond(dx,e,x0):
            return dx
    return Hfun,precond

def make_ham_func_twoSite(M,W,F,site,usePrecond=False,debug=False):
    # Define Hamiltonian function to give Hx
    (n1,n2,n3) = M[0].shape
    (n4,n5,n6) = M[1].shape
    def Hfun(x):
        x_reshape = np.reshape(x,(n2,n1,n4,n6)) # PH - Check ordering???
        fin_sum = np.zeros(x_reshape.shape,dtype=np.complex_)
        for mpoInd in range(len(W)):
            if W[mpoInd][site] is None:
            else:
                in_sum1 = einsum('ijk,lmnk->ijnml',F[mpoInd][1],x_reshape)
                in_sum2 = einsum('')
    return Hfun,precond

def make_ham_func(M,W,F,site,usePrecond=False,debug=False,oneSite=True):
    if oneSite:
        return make_ham_func_oneSite(M,W,F,site,usePrecond=usePrecond,debug=debug)
    else:
        return make_ham_func_twoSite(M,W,F,site,usePrecond=usePrecond,debug=debug)

def pick_eigs(w,v,nroots,x0):
    idx = np.argsort(np.real(w))
    w = w[idx]
    v = v[:,idx]
    return w, v, idx

def orthonormalize(vecs):
    _,nvecs = vecs.shape
    vecs[:,0] /= np.dot(vecs[:,0],np.conj(vecs[:,0]))
    for i in range(nvecs):
        proj = []
        for j in range(i):
            proj.append(np.dot(vecs[:,j],np.conj(vecs[:,i]))/np.dot(vecs[:,j],np.conj(vecs[:,j]))*vecs[:,j])
        for j in range(i):
            vecs[:,i] -= proj[j]
        vecs[:,i] /= np.dot(vecs[:,i],np.conj(vecs[:,i]))
    return vecs

def calc_eigs_exact(mpsL,W,F,site,
                    nStates,preserveState=False,
                    orthonormalize=False,oneSite=True):
    H = calc_ham(mpsL[0],W,F,site,oneSite=oneSite)
    Mprev = mpsL[0][site].ravel()
    vals,vecs = sla.eig(H)
    inds = np.argsort(vals)[::-1]
    E = vals[inds[:nStates]]
    vecs = vecs[:,inds[:nStates]]
    # Orthonormalize (when gap is too small?)
    # PH - Why is this needed?
    if (nStates > 1) and orthonormalize:
        vecs = sla.orth(vecs)
    # Don't preserve state at ends
    if (site == 0) or (site == len(mpsL[0])-1): preserveState = True
    E,vecs,ovlp = check_overlap(Mprev,vecs,E,preserveState=preserveState)
    return E,vecs,ovlp

def calc_eigs_arnoldi(mpsL,W,F,site,
                      nStates,nStatesCalc=None,
                      preserveState=False,orthonormalize=False,
                      oneSite=True):
    guess = np.reshape(mpsL[0][site],-1)
    Hfun,_ = make_ham_func(mpsL[0],W,F,site,oneSite=oneSite)
    (n1,n2,n3) = mpsL[0][site].shape
    H = LinearOperator((n1*n2*n3,n1*n2*n3),matvec=Hfun)
    if nStatesCalc is None: nStatesCalc = nStates
    nStates,nStatesCalc = min(nStates,n1*n2*n3-2), min(nStatesCalc,n1*n2*n3-2)
    try:
        vals,vecs = arnoldi(H,k=nStatesCalc,which='SR',v0=guess,tol=1e-5)
    except Exception as exc:
        vals = exc.eigenvalues
        vecs = exc.eigenvectors
    inds = np.argsort(vals)
    E = -vals[inds[:nStates]]
    vecs = vecs[:,inds[:nStates]]
    # Orthonormalize (when gap is too small?) - PH, Why?
    if (nStates > 1) and orthonormalize:
        vecs = sla.orth(vecs)
    # At the ends, we do not want to switch states when preserving state is off
    if (site == 0) or (site == len(mpsL[0])-1): preserveState = True
    E,vecs,ovlp = check_overlap(guess,vecs,E,preserveState=preserveState)
    return E,vecs,ovlp

def calc_eigs_davidson(mpsL,W,F,site,
                       nStates,nStatesCalc=None,
                       preserveState=False,orthonormalize=False,
                       oneSite=True):
    Hfun,precond = make_ham_func(mpsL[0],W,F,site,oneSite=oneSite)
    (n1,n2,n3) = mpsL[0][site].shape
    if nStatesCalc is None: nStatesCalc = nStates
    nStates,nStatesCalc = min(nStates,n1*n2*n3-1), min(nStatesCalc,n1*n2*n3-1)
    guess = []
    for state in range(nStates):
        guess.append(np.reshape(mpsL[state][site],-1))
    # PH - Could add some convergence check
    vals,vecso = davidson(Hfun,guess,precond,nroots=nStatesCalc,pick=pick_eigs,follow_state=False,tol=1e-10,max_cycle=100)
    sort_inds = np.argsort(np.real(vals))
    try:
        vecs = np.zeros((len(vecso[0]),nStates),dtype=np.complex_)
        E = -vals[sort_inds[:nStates]]
        for i in range(min(nStates,len(sort_inds))):
            vecs[:,i] = vecso[sort_inds[i]]
    except:
        vecs = vecso
        E = -vals
        pass
    # Allow orthonormalize to a float indicating a gap size where it should be turned on
    if not isinstance(orthonormalize,bool):
        if nStates == 1: orthonormalize = False
        else:
            gap  = np.abs(E[0]-E[1])
            if gap < orthonormalize:
                orthonormalize = True
            else:
                orthonormalize = False
    # Orthonormalize (when gap is too small?) - PH, Why?
    if (nStates > 1) and orthonormalize:
        vecs = sla.orth(vecs)
    # At the ends, we do not want to switch states when preserving state is off
    if (site == 0) or (site == len(mpsL[0])-1): preserveState = True
    E,vecs,ovlp = check_overlap(guess[0],vecs,E,preserveState=preserveState)
    return E,vecs,ovlp

def calc_eigs(mpsL,W,F,site,nStates,
              alg='davidson',preserveState=False,
              orthonormalize=False,oneSite=True):
    if alg == 'davidson':
        E,vecs,ovlp = calc_eigs_davidson(mpsL,W,F,site,nStates,
                                         preserveState=preserveState,
                                         orthonormalize=orthonormalize,
                                         oneSite=oneSite)
    elif alg == 'exact':
        E,vecs,ovlp = calc_eigs_exact(mpsL,W,F,site,nStates,
                                      preserveState=preserveState,
                                      orthonormalize=orthonormalize,
                                      oneSite=oneSite)
    elif alg == 'arnoldi':
        E,vecs,ovlp = calc_eigs_arnoldi(mpsL,W,F,site,nStates,
                                        preserveState=preserveState,
                                        orthonormalize=orthonormalize,
                                        oneSite=oneSite)
    return E,vecs,ovlp
