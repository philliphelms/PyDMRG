import numpy as np
import scipy.linalg as sla
from pyscf.lib import einsum
from pyscf.lib import eig as davidson
from scipy.sparse.linalg import eigs as arnoldi
from scipy.sparse.linalg import LinearOperator
from tools.mps_tools import *
import warnings

# To Do :
# - Left Eigenvectors
# - constant_mbd=True does not work

def make_mps_right(M):
    N = len(M)
    for i in range(int(N)-1,0,-1):
        M_reshape = np.swapaxes(M[i],0,1)
        (n1,n2,n3) = M_reshape.shape
        M_reshape = np.reshape(M_reshape,(n1,n2*n3))
        (U,s,V) = np.linalg.svd(M_reshape,full_matrices=False)
        M_reshape = np.reshape(V,(n1,n2,n3))
        M[i] = np.swapaxes(M_reshape,0,1)
        M[i-1] = einsum('klj,ji,i->kli',M[i-1],U,s)
    return M

def alloc_env(M,W,mbd):
    N = len(M)
    # Initialize Empty FL to hold all F lists
    env_lst = []
    for mpoInd in range(len(W)):
        if W[mpoInd][0] is not None:
            _,mbdW,_,_ = W[mpoInd][0].shape
        else: 
            mbdW = 1
        F = []
        F.append(np.array([[[1]]]))
        for site in range(int(N/2)):
            F.append(np.zeros((min(2**(site+1),mbd),mbdW,min(2**(site+1),mbd))))
        if N%2 is 1:
            F.append(np.zeros((min(2**(site+2),mbd),mbdW,min(2**(site+2),mbd))))
        for site in range(int(N/2)-1,0,-1):
            F.append(np.zeros((min(2**(site),mbd),mbdW,min(2**site,mbd))))
        F.append(np.array([[[1]]]))
        # Add environment to env list
        env_lst.append(F)
    return env_lst

def update_envL(M,W,F,site):
    for mpoInd in range(len(W)):
        if W[mpoInd][site] is None:
            tmp1 = einsum('eaf,cdf->eacd',M[site],F[mpoInd][site+1])
            F[mpoInd][site] = einsum('bacy,bxc->xya',tmp1,np.conj(M[site]))
        else:
            tmp1 = einsum('eaf,cdf->eacd',M[site],F[mpoInd][site+1])
            tmp2 = einsum('eacd,ydbe->acyb',tmp1,W[mpoInd][site])
            F[mpoInd][site] = einsum('acyb,bxc->xya',tmp2,np.conj(M[site]))
    return F

def update_envR(M,W,F,site):
    for mpoInd in range(len(W)):
        if W[mpoInd][site] is None:
            tmp1 = einsum('jlp,ijk->lpik',F[mpoInd][site],np.conj(M[site]))
            F[mpoInd][site+1] = einsum('npq,mpnk->kmq',M[site],tmp1)
        else:
            tmp1 = einsum('jlp,ijk->lpik',F[mpoInd][site],np.conj(M[site]))
            tmp2 = einsum('lmin,lpik->mpnk',W[mpoInd][site],tmp1)
            F[mpoInd][site+1] = einsum('npq,mpnk->kmq',M[site],tmp2)
    return F

def calc_env(M,W,mbd,gaugeSite=0):
    # PH - What to do with this gauge site stuff
    N = len(M)
    env_lst = alloc_env(M,W,mbd)
    # Calculate Environment From Right
    for site in range(int(N)-1,gaugeSite,-1):
        env_lst = update_envL(M,W,env_lst,site)
    # Calculate Environment from Left
    for site in range(gaugeSite):
        env_lst = update_envR(M,W,env_lst,site)
    return env_lst

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

def calcRDM(M,swpDir):
    if swpDir == 'right':
        (n1,n2,n3) = M.shape
        M = np.swapaxes(M,0,1)
        M = np.reshape(M,(n2*n1,n3))
        return einsum('ij,kj->ik',M,np.conj(M))
    elif swpDir == 'left':
        (n1,n2,n3) = M.shape
        M = np.swapaxes(M,0,1)
        M = np.reshape(M,(n2,n1*n3))
        return einsum('ij,ik->jk',M,np.conj(M))

def calc_ent_right(M,v,site):
    (n1,n2,n3) = M[site].shape
    Mtmp = np.reshape(v,(n1,n2,n3))
    M_reshape = np.reshape(Mtmp,(n1*n2,n3))
    (_,S,_) = np.linalg.svd(M_reshape,full_matrices=False)
    EE,EEs = calc_entanglement(S)
    print('\t\tEE = {}'.format(EE))
    return EE, EEs

def calc_ent_left(M,v,site):
    (n1,n2,n3) = M[site].shape
    Mtmp = np.reshape(v,(n1,n2,n3))
    M_reshape = np.swapaxes(Mtmp,0,1)
    M_reshape = np.reshape(M_reshape,(n2,n1*n3))
    (_,S,_) = np.linalg.svd(M_reshape,full_matrices=False)
    EE,EEs = calc_entanglement(S)
    print('\t\tEE = {}'.format(EE))
    return EE, EEs

def renormalizeR(M,v,site,nStates=1,targetState=0):
    (n1,n2,n3) = M[site].shape
    # Try to calculate Entanglement?
    EE,EEs = calc_ent_right(M,v[:,targetState],site)
    # Calculate the reduced density matrix
    _,nStatesCalc = v.shape
    nStates = min(nStates,nStatesCalc)
    for i in range(nStates):
        if nStates != 1: 
            vtmp = v[:,i]
        else:
            vtmp  = v
        vReshape = np.reshape(vtmp,(n1,n2,n3))
        w = 1./float(nStates)
        if i == 0:
            rdm = w*calcRDM(vReshape,'right')
        else:
            rdm +=w*calcRDM(vReshape,'right')
    # Take eigenvalues of the rdm
    vals,vecs = np.linalg.eig(rdm)
    # Sort Inds
    inds = np.argsort(vals)[::-1]
    # Keep only maxBondDim eigenstates
    inds = inds[:n3]
    vals = vals[inds]
    vecs = vecs[:,inds]
    # Make sure vecs are orthonormal
    vecs = sla.orth(vecs)
    # Put resulting vectors into MPS
    M[site] = np.reshape(vecs,(n2,n1,n3))
    M[site] = np.swapaxes(M[site],0,1)
    if not np.all(np.isclose(einsum('ijk,ijl->kl',M[site],np.conj(M[site])),np.eye(n3),atol=1e-6)):
        print('\t\tNormalization Problem')
    # Calculate next site for guess
    if nStates != 1:
        vReshape = np.reshape(v[:,targetState],(n1,n2,n3))
    else:
        vReshape = np.reshape(v,(n1,n2,n3))
    # PH - This next line is incorrect!!!
    M[site+1] = einsum('lmn,lmk,ikj->inj',np.conj(M[site]),vReshape,M[site+1])
    return M,EE,EEs

def renormalizeL(M,v,site,nStates=1,targetState=0):
    (n1,n2,n3) = M[site].shape
    # Try to calculate Entanglement?
    EE,EEs = calc_ent_left(M,v[:,targetState],site)
    # Calculate the reduced density matrix
    _,nStatesCalc = v.shape
    nStates = min(nStates,nStatesCalc)
    for i in range(nStates):
        if nStates != 1: 
            vtmp = v[:,i]
        else: 
            vtmp = v
        vReshape = np.reshape(vtmp,(n1,n2,n3))
        w = 1./float(nStates)
        if i == 0:
            rdm = w*calcRDM(vReshape,'left') 
        else:
            rdm +=w*calcRDM(vReshape,'left')
    # Take eigenvalues of the rdm
    vals,vecs = np.linalg.eig(rdm) # Transpose here is useless...
    # Sort inds
    inds = np.argsort(vals)[::-1]
    # Keep only maxBondDim eigenstates
    inds = inds[:n2]
    vals = vals[inds]
    vecs = vecs[:,inds]
    # Make sure vecs are orthonormal
    vecs = sla.orth(vecs)
    vecs = vecs.T
    # Put resulting vectors into MPS
    M[site] = np.reshape(vecs,(n2,n1,n3))
    M[site] = np.swapaxes(M[site],0,1)
    if not np.all(np.isclose(einsum('ijk,ilk->jl',M[site],np.conj(M[site])),np.eye(n2),atol=1e-6)):
        print('\t\tNormalization Problem')
    # Calculate next site's guess
    if nStates != 1:
        vReshape = np.reshape(v[:,targetState],(n1,n2,n3))
    else:
        vReshape = np.reshape(v,(n1,n2,n3))
    M[site-1] = einsum('ijk,lkm,lnm->ijn',M[site-1],vReshape,np.conj(M[site]))
    return M,EE,EEs

def check_overlap(Mprev,vecs,E,preserveState=False,printStates=False,allowSwap=True):
    _,nVecs = vecs.shape
    # Check to make sure we dont have a small gap
    reuseOldState = True
    for j in range(nVecs):
        ovlp_j = np.abs(np.dot(Mprev,np.conj(vecs[:,j])))
        print('\t\tChecking Overlap {} = {}'.format(j,ovlp_j))
        if ovlp_j > 0.98:
            reuseOldState = False
            if (j != 0) and preserveState and allowSwap:
                # Swap eigenstates
                print('!!! Swapping States {} & {} !!!'.format(0,j))
                tmpVec = vecs[:,j]
                vecs[:,j] = vecs[:,0]
                vecs[:,0] = tmpVec
                Etmp = E[j]
                E[j] = E[0]
                E[0] = Etmp
    if reuseOldState and preserveState:
        print('!!! Correct State Not Found !!!')
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

def make_ham_func(M,W,F,site,usePrecond=False,debug=False):
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
    # Test if they are really orthonormal:
    #print('Checking orthonormalization:')
    #for i in range(nvecs):
    #    for j in range(nvecs):
    #        print('u({})*u({}) = {}'.format(i,j,np.dot(vecs[:,i],np.conj(vecs[:,j]))))
    return vecs

def calc_eigs_exact(M,W,F,site,nStates,preserveState=False):
    H = calc_ham(M,W,F,site)
    Mprev = M[site].ravel()
    vals,vecs = sla.eig(H)
    inds = np.argsort(vals)[::-1]
    E = vals[inds[:nStates]]
    vecs = vecs[:,inds[:nStates]]
    # Check if lowest two states are degenerate
    if nStates > 1:
        gap = np.abs(E[0]-E[1])
        #vecs = orthonormalize(vecs)
        if gap < 1e-8:
            print('ORTHONORMALIZING')
            vecs = sla.orth(vecs)
    # Don't preserve state at ends
    if not preserveState:
        if (site == 0) or (site == len(M)-1):
            preserveState = True
    E,vecs,ovlp = check_overlap(Mprev,vecs,E,preserveState=preserveState)
    print(E)
    print(vecs.shape)
    return E,vecs,ovlp

def calc_eigs_arnoldi(M,W,F,site,nStates,nStatesCalc=None,preserveState=False):
    guess = np.reshape(M[site],-1)
    Hfun,_ = make_ham_func(M,W,F,site)
    (n1,n2,n3) = M[site].shape
    H = LinearOperator((n1*n2*n3,n1*n2*n3),matvec=Hfun)
    if nStatesCalc is None: nStatesCalc = nStates+2
    nStates,nStatesCalc = min(nStates,n1*n2*n3-2), min(nStatesCalc,n1*n2*n3-2)
    try:
        vals,vecs = arnoldi(H,k=nStatesCalc,which='SR',v0=guess,tol=1e-5)
    except Exception as exc:
        vals = exc.eigenvalues
        vecs = exc.eigenvectors
    inds = np.argsort(vals)
    E = -vals[inds[:nStates]]
    vecs = vecs[:,inds[:nStates]]
    # At the ends, we do not want to switch states when preserving state is off
    if not preserveState:
        if (site == 0) or (site == len(M)-1):
            preserveState = True
    E,vecs,ovlp = check_overlap(guess,vecs,E,preserveState=preserveState)
    return E,vecs,ovlp

def calc_eigs_davidson(M,W,F,site,nStates,nStatesCalc=None,preserveState=False):
    Hfun,precond = make_ham_func(M,W,F,site)
    (n1,n2,n3) = M[site].shape
    if nStatesCalc is None: nStatesCalc = nStates+2
    nStates,nStatesCalc = min(nStates,n1*n2*n3-1), min(nStatesCalc,n1*n2*n3-1)
    guess = []
    for i in range(nStatesCalc):
        if i == 0:
            guess.append(np.reshape(M[site],-1))
        else:
            guess.append(np.random.rand(len(np.reshape(M[site],-1)))+1j*np.zeros((len(np.reshape(M[site],-1)))))
    vals,vecso = davidson(Hfun,guess,precond,nroots=nStatesCalc,pick=pick_eigs,follow_state=False,tol=1e-16)
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
    E,vecs,ovlp = check_overlap(guess[0],vecs,E,preserveState=preserveState)
    return E,vecs,ovlp

def calc_eigs(M,W,F,site,nStates,alg='arnoldi',preserveState=False):
    if alg == 'davidson':
        E,vecs,ovlp = calc_eigs_davidson(M,W,F,site,nStates,preserveState=preserveState)
    elif alg == 'exact':
        E,vecs,ovlp = calc_eigs_exact(M,W,F,site,nStates,preserveState=preserveState)
    elif alg == 'arnoldi':
        E,vecs,ovlp = calc_eigs_arnoldi(M,W,F,site,nStates,preserveState=preserveState)
    return E,vecs,ovlp

def calc_entanglement(S):
    # Ensure correct normalization
    S /= np.sqrt(np.dot(S,np.conj(S)))
    assert(np.isclose(np.abs(np.sum(S*np.conj(S))),1.))
    EEspec = -S*np.conj(S)*np.log2(S*np.conj(S))
    EE = np.sum(EEspec)
    return EE,EEspec

def rightStep(M,W,F,site,nStates=1,alg='arnoldi',preserveState=False):
    E,v,ovlp = calc_eigs(M,W,F,site,nStates,alg=alg,preserveState=preserveState)
    M,EE,EEs = renormalizeR(M,v,site,nStates=nStates)
    F = update_envR(M,W,F,site)
    return E,M,F,EE,EEs

def rightSweep(M,W,F,iterCnt,nStates=1,alg='arnoldi',preserveState=False,startSite=None,endSite=None):
    N = len(M)
    if startSite is None: startSite = 0
    if endSite is None: endSite = N-1
    Ereturn = None
    EE = None
    EEs = None
    print('Right Sweep {}'.format(iterCnt))
    for site in range(startSite,endSite):
        E,M,F,_EE,_EEs = rightStep(M,W,F,site,nStates,alg=alg,preserveState=preserveState)
        print('\tEnergy at Site {}: {}'.format(site,E))
        if site == int(N/2):
            Ereturn = E
            EE = _EE
            EEs= _EEs
    return Ereturn,M,F,EE,EEs

def leftStep(M,W,F,site,nStates=1,alg='arnoldi',preserveState=False):
    E,v,ovlp = calc_eigs(M,W,F,site,nStates,alg=alg,preserveState=preserveState)
    M,EE,EEs = renormalizeL(M,v,site,nStates=nStates)
    F = update_envL(M,W,F,site)
    return E,M,F,EE,EEs

def leftSweep(M,W,F,iterCnt,nStates=1,alg='arnoldi',preserveState=False,startSite=None,endSite=None):
    N = len(M)
    if startSite is None: startSite = N-1
    if endSite is None: endSite = 0
    Ereturn = None
    EE = None
    EEs = None
    print('Left Sweep {}'.format(iterCnt))
    for site in range(startSite,endSite,-1):
        E,M,F,_EE,_EEs = leftStep(M,W,F,site,nStates,alg=alg,preserveState=preserveState)
        print('\tEnergy at Site {}: {}'.format(site,E))
        if site == int(N/2):
            Ereturn = E
            EE = _EE
            EEs= _EEs
    return Ereturn,M,F,EE,EEs

def checkConv(E_prev,E,tol,iterCnt,maxIter,minIter,nStates=1,targetState=0,EE=None,EEspec=[None]):
    if nStates != 1: E = E[targetState]
    if (np.abs(E-E_prev) < tol) and (iterCnt > minIter):
        cont = False
        conv = True
    elif iterCnt > maxIter:
        cont = False
        conv = False
    else:
        iterCnt += 1
        E_prev = E
        cont = True
        conv = False
    return cont,conv,E_prev,iterCnt

def observable_sweep(M,F):
    # Going to the right
    # PH - Only calculates Entanglement Currently
    # PH - Not in use now
    N = len(M)
    for site in range(N-1):
        (n1,n2,n3) = M[site].shape
        M_reshape = np.reshape(M[site],(n1*n2,n3))
        (U,S,V) = np.linalg.svd(M_reshape,full_matrices=False)
        M[site] = np.reshape(U,(n1,n2,n3))
        M[site+1] = einsum('i,ij,kjl->kil',S,V,M[site+1])
        if site == int(N/2):
            EE,EEs = calc_entanglement(S)
    return EE,EEs

def printResults(converged,E,EE,EEspec,gap):
    print('#'*75)
    if converged:
        print('Converged at E = {}'.format(E))
    else:
        print('Convergence not acheived, E = {}'.format(E))
    print('\tGap = {}'.format(gap))
    print('\tEntanglement Entropy  = {}'.format(EE))
    print('\tEntanglement Spectrum =')
    for i in range(len(EEspec)):
        print('\t\t{}'.format(EEspec[i]))
    print('#'*75)

def run_sweeps(M,W,F,initGuess=None,maxIter=0,minIter=None,
               tol=1e-5,fname = None,nStates=1,
               targetState=0,alg='arnoldi',
               preserveState=False,gaugeSiteLoad=0,
               gaugeSiteSave=0,returnState=False,
               returnEnv=False,returnEntSpec=False):
    cont = True
    iterCnt = 0
    E_prev = 0
    if gaugeSiteLoad != 0:
        E,M,F,EE,EEs = rightSweep(M,W,F,iterCnt,nStates=nStates,alg=alg,preserveState=preserveState,startSite=gaugeSiteLoad)
        E,M,F,EE,EEs = leftSweep(M,W,F,iterCnt,nStates=nStates,alg=alg,preserveState=preserveState)
    while cont:
        if iterCnt > 2: preserveState = True # PH - Added this line experimentally
        E,M,F,EE,EEs = rightSweep(M,W,F,iterCnt,nStates=nStates,alg=alg,preserveState=preserveState)
        E,M,F,EE,EEs = leftSweep(M,W,F,iterCnt,nStates=nStates,alg=alg,preserveState=preserveState)
        cont,conv,E_prev,iterCnt = checkConv(E_prev,E,tol,iterCnt,maxIter,minIter,nStates=nStates,targetState=targetState)
    if gaugeSiteSave != 0:
        _E,M,F,_EE,_EEs = rightSweep(M,W,F,iterCnt+1,nStates=nStates,alg=alg,preserveState=preserveState,endSite=gaugeSiteSave)
        if _E is not None:
            E,EE,EEs = _E,_EE,_EEs
    save_mps(M,fname,gaugeSite=gaugeSiteSave)
    #EE,EEs = observable_sweep(M,F)
    if nStates != 1: 
        gap = E[0]-E[1]
    else:
        gap = None
    if hasattr(E,'__len__'): E = E[targetState]
    printResults(conv,E,EE,EEs,gap)
    output = [E,EE,gap]
    if returnEntSpec:
        output.append(EEs)
    if returnState:
        output.append(M)
    if returnEnv:
        output.append(F)
    return output

def run_dmrg(mpo,env=None,initGuess=None,mbd=[2,4,8,16],
             tol=1e-5,maxIter=10,minIter=0,fname=None,
             nStates=1,targetState=0,
             constant_mbd=False,alg='arnoldi',
             preserveState=False,gaugeSiteSave=None,
             returnState=False,returnEnv=False,returnEntSpec=False):
    N = len(mpo[0])
    if gaugeSiteSave is None: gaugeSiteSave = int(N/2)+1
    # Make sure everything is a vector
    if not hasattr(mbd,'__len__'): mbd = np.array([mbd])
    if not hasattr(tol,'__len__'):
        tol = tol*np.ones(len(mbd))
    else:
        assert(len(mbd) == len(tol))
    if not hasattr(maxIter,'__len__'):
        maxIter = maxIter*np.ones(len(mbd))
    else:
        assert(len(maxIter) == len(mbd))
    if not hasattr(minIter,'__len__'):
        minIter = minIter*np.ones(len(mbd))
    else:
        assert(len(minIter) == len(mbd))
    # Vector to store results
    Evec  = np.zeros(len(mbd),dtype=np.complex_)
    EEvec = np.zeros(len(mbd),dtype=np.complex_)
    gapvec= np.zeros(len(mbd),dtype=np.complex_)
    for mbdInd,mbdi in enumerate(mbd):
        print('Starting Calc for MBD = {}'.format(mbdi))
        if initGuess is None:
            mps = create_rand_mps(N,mbdi)
            mps = make_mps_right(mps)
            if constant_mbd: mps = increase_mbd(mps,mbdi,constant=True)
            gSite = 0
        else:
            if mbdInd == 0:
                mps,gSite = load_mps(N,initGuess+'_mbd'+str(mbdInd))
            else:
                mps,gSite = load_mps(N,initGuess+'_mbd'+str(mbdInd-1))
                mps = increase_mbd(mps,mbdi)
        if env is None: env = calc_env(mps,mpo,mbdi,gaugeSite=gSite)
        fname_tmp = None
        if fname is not None: fname_tmp = fname + '_mbd' + str(mbdInd)
        output = run_sweeps(mps,mpo,env,
                              maxIter=maxIter[mbdInd],
                              minIter=minIter[mbdInd],
                              tol=tol[mbdInd],
                              fname=fname_tmp,
                              nStates=nStates,
                              alg=alg,
                              targetState=targetState,
                              gaugeSiteLoad=gSite,
                              gaugeSiteSave=gaugeSiteSave,
                              preserveState=preserveState,
                              returnState=returnState,
                              returnEnv=returnEnv,
                              returnEntSpec=returnEntSpec)
        # Extract Results
        E = output[0]
        EE = output[1]
        gap = output[2]
        Evec[mbdInd]  = output[0]
        EEvec[mbdInd] = output[1]
        gapvec[mbdInd]= output[2]
        if returnEntSpec and returnState and returnEnv:
            EEs = output[3]
            mps = output[4]
            env = output[5]
        elif returnEntSpec and returnState:
            EEs = output[3]
            mps = output[4]
        elif returnEntSpec and returnEnv:
            EEs = output[3]
            env = output[4]
        elif returnState and returnEnv:
            mps = output[3]
            env = output[4]
        elif returnEntSpec:
            EEs = output[3]
        elif returnState:
            mps = output[3]
        elif returnEnv:
            env = output[3]
    # Return Results
    if len(Evec) == 1:
        output = [E,EE,gap]
    else:
        output = [Evec,EEvec,gapvec]
    if returnEntSpec:
        output.append(EEs)
    if returnState:
        output.append(mps)
    if returnEnv:
        output.append(env)
    return output
