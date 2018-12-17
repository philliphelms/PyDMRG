import numpy as np
import scipy.linalg as sla
from pyscf.lib import einsum
from pyscf.lib import eig as davidson
from scipy.sparse.linalg import eigs as arnoldi
from scipy.sparse.linalg import LinearOperator

# To Do :
# - Left Eigenvectors
# - constant_mbd=True does not work

def increase_mbd(M,mbd,periodic=False,constant=False,d=2):
    N = len(M)
    if periodic == False:
        if constant == False:
            for site in range(int(N/2)):
                nx,ny,nz = M[site].shape
                sz1 = min(d**site,mbd)
                sz2 = min(d**(site+1),mbd)
                M[site] = np.pad(M[site], ((0,0), (0,sz1-ny), (0,sz2-nz)), 'constant', constant_values=0j)
            if N%2 is 1:
                site += 1
                nx,ny,nz = M[site].shape
                sz1 = min(d**(site),mbd)
                sz2 = min(d**(site),mbd)
                M[site] = np.pad(M[site], ((0,0), (0,sz1-ny), (0,sz2-nz)), 'constant', constant_values=0j)
            for i in range(int(N/2))[::-1]:
                site = N - i - 1
                nx,ny,nz = M[site].shape
                sz1 = min(d**(N-(site)),mbd)
                sz2 = min(d**(N-(site+1)),mbd)
                M[site] = np.pad(M[site], ((0,0), (0,sz1-ny), (0,sz2-nz)), 'constant', constant_values=0j)
        else:
            for site in range(N):
                nx,ny,nz = M[site].shape
                sz1 = mbd
                sz2 = mbd
                if site == 0: sz1 = 1
                if site == N-1: sz2 = 1
                M[site] = np.pad(M[site], ((0,0), (0,sz1-ny), (0,sz2-nz)), 'constant', constant_values=0j)
    else:
        for site in range(N):
            nx,ny,nz = M[site].shape
            sz1 = mbd
            sz2 = mbd
            M[site] = np.pad(M[site], ((0,0), (0,sz1-ny), (0,sz2-nz)), 'constant', constant_values=0j)
    return M

def create_rand_mps(N,mbd,d=2):
    # Create MPS
    M = []
    for i in range(int(N/2)):
        #M.insert(len(M),np.random.rand(d,min(d**(i),mbd),min(d**(i+1),mbd)))
        M.insert(len(M),np.ones((d,min(d**(i),mbd),min(d**(i+1),mbd))))
    if N%2 is 1:
        #M.insert(len(M),np.random.rand(d,min(d**(i+1),mbd),min(d**(i+1),mbd)))
        M.insert(len(M),np.ones((d,min(d**(i+1),mbd),min(d**(i+1),mbd))))
    for i in range(int(N/2))[::-1]:
        #M.insert(len(M),np.random.rand(d,min(d**(i+1),mbd),min(d**i,mbd)))
        M.insert(len(M),np.ones((d,min(d**(i+1),mbd),min(d**i,mbd))))
    return M

def load_mps(N,fname):
    npzfile = np.load(fname+'.npz')
    M = []
    for i in range(N):
        M.append(npzfile['M'+str(i)])
    return M

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

def calc_env(M,W,mbd):
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
    # Calculate Environment
    for site in range(int(N)-1,0,-1):
        for mpoInd in range(len(W)):
            if W[mpoInd][site] is None:
                tmp1 = einsum('eaf,cdf->eacd',M[site],env_lst[mpoInd][site+1])
                env_lst[mpoInd][site] = einsum('bacy,bxc->xya',tmp1,np.conj(M[site]))
            else:
                tmp1 = einsum('eaf,cdf->eacd',M[site],env_lst[mpoInd][site+1])
                tmp2 = einsum('eacd,ydbe->acyb',tmp1,W[mpoInd][site])
                env_lst[mpoInd][site] = einsum('acyb,bxc->xya',tmp2,np.conj(M[site]))
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

def renormalizeR(M,v,site,nStates=1,targetState=0):
    (n1,n2,n3) = M[site].shape
    # Calculate the reduced density matrix
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
    M[site+1] = einsum('ijk,ijl,mkn->mln',np.conj(M[site]),vReshape,M[site+1])
    return M

def renormalizeL(M,v,site,nStates=1,targetState=0):
    (n1,n2,n3) = M[site].shape
    # Calculate the reduced density matrix
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
    return M

def calc_eigs_exact(M,W,F,site,nStates):
    H = calc_ham(M,W,F,site)
    Mprev = M[site].ravel()
    vals,vecs = sla.eig(H)
    inds = np.argsort(vals)[::-1]
    E = vals[inds[:nStates]]
    vecs = vecs[:,inds[:nStates]]
    return E,vecs

def make_ham_func(M,W,F,site):
    H = calc_ham(M,W,F,site)
    def Hfun(x):
        x_reshape = np.reshape(x,M[site].shape)
        fin_sum = np.zeros(x_reshape.shape,dtype=np.complex_)
        for mpoInd in range(len(W)):
            if W[mpoInd][site] is None:
                in_sum1 = einsum('ijk,lmk->ijlm',F[mpoInd][site+1],x_reshape)
                fin_sum +=einsum('pnm,inom->opi',F[mpoInd][site],in_sum1)
            else:
                in_sum1 = einsum('ijk,lmk->ijlm',F[mpoInd][site+1],x_reshape)
                in_sum2 = einsum('njol,ijlm->noim',W[mpoInd][site],in_sum1)
                fin_sum +=einsum('pnm,noim->opi',F[mpoInd][site],in_sum2)
        assert(np.isclose(np.sum(np.abs(np.reshape(fin_sum,-1)-np.dot(H,x))),0))
        return -np.reshape(fin_sum,-1)
    diag = calc_diag(M,W,F,site)
    assert(np.isclose(np.sum(np.abs(diag-np.diag(H))),0))
    def precond(dx,e,x0):
        return dx/diag-e
    return Hfun,precond

def pick_eigs(w,v,nroots,x0):
    idx = w.real.argsort()
    return w[idx], v[:,idx],idx

def calc_eigs_davidson(M,W,F,site,nStates):
    H,precond = make_ham_func(M,W,F,site)
    guess = np.reshape(M[site],-1)
    vals,vecso = davidson(H,guess,precond,nroots=nStates,pick=pick_eigs,follow_state=True,tol=1e-8)
    sort_inds = np.argsort(np.real(vals))
    try:
        vecs = np.zeros((len(vecso[0]),nStates),dtype=np.complex_)
        vals = -vals[sort_inds[:nStates]]
        for i in range(nStates):
            vecs[:,i] = vecso[sort_inds[i]]
    except:
        vecs = vecso
        pass
    return vals,vecs

def calc_eigs_arnoldi(M,W,F,site,nStates):
    guess = np.reshape(M[site],-1)
    Hfun,_ = make_ham_func(M,W,F,site)
    (n1,n2,n3) = M[site].shape
    H = LinearOperator((n1*n2*n3,n1*n2*n3),matvec=Hfun)
    try:
        vals,vecs = arnoldi(H,k=nStates,which='SR',v0=guess,tol=1e-5)
    except Exception as exc:
        vals = exc.eigenvalues
        vecs = exc.eigenvectors
    inds = np.argsort(vals)
    E = -vals[inds[:nStates]]
    vecs = vecs[:,inds[:nStates]]
    return E,vecs

def calc_eigs(M,W,F,site,nStates,alg='exact'):
    if alg == 'davidson':
        E,vecs = calc_eigs_davidson(M,W,F,site,nStates)
    elif alg == 'exact':
        E,vecs = calc_eigs_exact(M,W,F,site,nStates)
    elif alg == 'arnoldi':
        E,vecs = calc_eigs_arnoldi(M,W,F,site,nStates)
    return E,vecs

def calc_entanglement(S):
    EEspec = -S**2.*np.log2(S**2.)
    EE = np.sum(EEspec)
    return EE,EEspec

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

def rightSweep(M,W,F,iterCnt,nStates=1,alg='exact'):
    N = len(M)
    print('Right Sweep {}'.format(iterCnt))
    for site in range(N-1):
        #diag = calc_diag(M,W,F,site)
        E,v = calc_eigs(M,W,F,site,nStates,alg=alg)
        M = renormalizeR(M,v,site,nStates=nStates)
        F = update_envR(M,W,F,site)
        print('\tEnergy at Site {}: {}'.format(site,E))
        if site == int(N/2):
            Ereturn = E
    return Ereturn,M,F

def leftSweep(M,W,F,iterCnt,nStates=1,alg='exact'):
    N = len(M)
    print('Left Sweep {}'.format(iterCnt))
    for site in range(N-1,0,-1):
        #diag = calc_diag(M,W,F,site)
        E,v = calc_eigs(M,W,F,site,nStates,alg=alg)
        M = renormalizeL(M,v,site,nStates=nStates)
        F = update_envL(M,W,F,site)
        print('\tEnergy at Site {}: {}'.format(site,E))
        if site == int(N/2):
            Ereturn = E
    return Ereturn,M,F

def checkConv(E_prev,E,tol,iterCnt,maxIter,nStates=1,targetState=0):
    if nStates != 1: E = E[targetState]
    if np.abs(E-E_prev) < tol:
        print('#'*75+'\nConverged at E = {}'.format(E)+'\n'+'#'*75)
        converged = True
    elif iterCnt > maxIter:
        print('Convergence not acheived')
        converged = True
    else:
        iterCnt += 1
        E_prev = E
        converged = False
    return converged,E_prev,iterCnt

def save_mps(M,fname):
    if fname is not None:
        Mdict = {}
        for i in range(len(M)):
            Mdict['M'+str(i)] = M[i]
        np.savez(fname+'.npz',**Mdict)

def observable_sweep(M,F):
    # Going to the right
    # PH - Only calculates Entanglement Currently
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

def run_sweeps(M,W,F,initGuess=None,maxIter=10,tol=1e-5,fname = None,nStates=1,targetState=0,alg='exact'):
    converged = False
    iterCnt = 0
    E_prev = 0
    while not converged:
        E,M,F = rightSweep(M,W,F,iterCnt,nStates=nStates,alg=alg)
        E,M,F = leftSweep(M,W,F,iterCnt,nStates=nStates,alg=alg)
        converged,E_prev,iterCnt = checkConv(E_prev,E,tol,iterCnt,maxIter,nStates=nStates,targetState=targetState)
    save_mps(M,fname)
    EE,EEs = observable_sweep(M,F)
    if nStates != 1: 
        gap = E[0]-E[1]
    else:
        gap = None
    if hasattr(E,'__len__'): E = E[targetState]
    return E,EE,gap

def run_dmrg(mpo,initGuess=None,mbd=[2,4,8,16],tol=1e-5,maxIter=3,fname=None,nStates=1,targetState=0,constant_mbd=False,alg='exact'):
    N = len(mpo[0])
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
        else:
            if mbdInd == 0:
                mps = load_mps(N,initGuess+'_mbd'+str(mbdInd))
            else:
                mps = load_mps(N,initGuess+'_mbd'+str(mbdInd-1))
                mps = increase_mbd(mps,mbdi)
        env = calc_env(mps,mpo,mbdi)
        fname_tmp = None
        if fname is not None: fname_tmp = fname + '_mbd' + str(mbdInd)
        E,EE,gap = run_sweeps(mps,mpo,env,
                             maxIter=maxIter[mbdInd],
                             tol=tol[mbdInd],
                             fname=fname_tmp,
                             nStates=nStates,
                             alg=alg,
                             targetState=targetState)
        Evec[mbdInd]  = E
        EEvec[mbdInd] = EE
        gapvec[mbdInd]=gap
    return Evec,EEvec,gapvec
