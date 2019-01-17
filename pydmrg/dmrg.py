import numpy as np
import scipy.linalg as sla
from pyscf.lib import eig as davidson
from pyscf.lib import einsum
from scipy.sparse.linalg import eigs as arnoldi
from scipy.sparse.linalg import LinearOperator
from tools.mps_tools import *
from tools.diag_tools import *
import warnings

# To Do :
# - Left Eigenvectors
# - constant_mbd=True does not work

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

def renormalizeR(mpsL,v,site,nStates=1,targetState=0):
    (n1,n2,n3) = mpsL[0][site].shape
    # Try to calculate Entanglement?
    EE,EEs = calc_ent_right(mpsL[0],v[:,targetState],site)
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
    # Loop through all MPS in list
    for state in range(nStates):
        # Put resulting vectors into MPS
        mpsL[state][site] = np.reshape(vecs,(n2,n1,n3))
        mpsL[state][site] = np.swapaxes(mpsL[state][site],0,1)
        if not np.all(np.isclose(einsum('ijk,ijl->kl',mpsL[state][site],np.conj(mpsL[state][site])),np.eye(n3),atol=1e-6)):
            print('\t\tNormalization Problem')
        # Calculate next site for guess
        if nStates != 1:
            vReshape = np.reshape(v[:,state],(n1,n2,n3))
        else:
            vReshape = np.reshape(v,(n1,n2,n3))
        # PH - This next line is incorrect!!!
        mpsL[state][site+1] = einsum('lmn,lmk,ikj->inj',np.conj(mpsL[state][site]),vReshape,mpsL[state][site+1])
    return mpsL,EE,EEs

def renormalizeL(mpsL,v,site,nStates=1,targetState=0):
    (n1,n2,n3) = mpsL[0][site].shape
    # Try to calculate Entanglement?
    EE,EEs = calc_ent_left(mpsL[0],v[:,targetState],site)
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
    # Loops through all MPS in list
    for state in range(nStates):
        # Put resulting vectors into MPS
        mpsL[state][site] = np.reshape(vecs,(n2,n1,n3))
        mpsL[state][site] = np.swapaxes(mpsL[state][site],0,1)
        if not np.all(np.isclose(einsum('ijk,ilk->jl',mpsL[state][site],np.conj(mpsL[state][site])),np.eye(n2),atol=1e-6)):
            print('\t\tNormalization Problem')
        # Calculate next site's guess
        if nStates != 1:
            vReshape = np.reshape(v[:,state],(n1,n2,n3))
        else:
            vReshape = np.reshape(v,(n1,n2,n3))
        # Push gauge onto next site
        mpsL[state][site-1] = einsum('ijk,lkm,lnm->ijn',mpsL[state][site-1],vReshape,np.conj(mpsL[state][site]))
    return mpsL,EE,EEs

def calc_entanglement(S):
    # Ensure correct normalization
    S /= np.sqrt(np.dot(S,np.conj(S)))
    assert(np.isclose(np.abs(np.sum(S*np.conj(S))),1.))
    EEspec = -S*np.conj(S)*np.log2(S*np.conj(S))
    EE = np.sum(EEspec)
    return EE,EEspec

def rightStep(mpsL,W,F,site,
              nStates=1,alg='arnoldi',
              preserveState=False,orthonormalize=False):
    E,v,ovlp = calc_eigs(mpsL,W,F,site,
                         nStates,
                         alg=alg,
                         preserveState=preserveState,
                         orthonormalize=orthonormalize)
    mpsL,EE,EEs = renormalizeR(mpsL,v,site,nStates=nStates)
    F = update_envR(mpsL[0],W,F,site)
    return E,mpsL,F,EE,EEs

def rightSweep(mpsL,W,F,iterCnt,
               nStates=1,alg='arnoldi',
               preserveState=False,startSite=None,
               endSite=None,orthonormalize=False):
    N = len(mpsL[0])
    if startSite is None: startSite = 0
    if endSite is None: endSite = N-1
    Ereturn = None
    EE = None
    EEs = None
    print('Right Sweep {}'.format(iterCnt))
    for site in range(startSite,endSite):
        E,mpsL,F,_EE,_EEs = rightStep(mpsL,W,F,site,
                                      nStates,
                                      alg=alg,
                                      preserveState=preserveState,
                                      orthonormalize=orthonormalize)
        print('\tEnergy at Site {}: {}'.format(site,E))
        if site == int(N/2):
            Ereturn = E
            EE = _EE
            EEs= _EEs
    return Ereturn,mpsL,F,EE,EEs

def leftStep(mpsL,W,F,site,
             nStates=1,alg='arnoldi',
             preserveState=False,orthonormalize=False):
    E,v,ovlp = calc_eigs(mpsL,W,F,site,
                         nStates,
                         alg=alg,
                         preserveState=preserveState,
                         orthonormalize=orthonormalize)
    mpsL,EE,EEs = renormalizeL(mpsL,v,site,nStates=nStates)
    F = update_envL(mpsL[0],W,F,site)
    return E,mpsL,F,EE,EEs

def leftSweep(mpsL,W,F,iterCnt,
              nStates=1,alg='arnoldi',
              preserveState=False,startSite=None,
              endSite=None,orthonormalize=False):
    N = len(mpsL[0])
    if startSite is None: startSite = N-1
    if endSite is None: endSite = 0
    Ereturn = None
    EE = None
    EEs = None
    print('Left Sweep {}'.format(iterCnt))
    for site in range(startSite,endSite,-1):
        E,mpsL,F,_EE,_EEs = leftStep(mpsL,W,F,site,
                                     nStates,
                                     alg=alg,
                                     preserveState=preserveState,
                                     orthonormalize=orthonormalize)
        print('\tEnergy at Site {}: {}'.format(site,E))
        if site == int(N/2):
            Ereturn = E
            EE = _EE
            EEs= _EEs
    return Ereturn,mpsL,F,EE,EEs

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

def run_sweeps(mpsL,W,F,initGuess=None,maxIter=0,minIter=None,
               tol=1e-5,fname = None,nStates=1,
               targetState=0,alg='arnoldi',
               preserveState=False,gaugeSiteLoad=0,
               gaugeSiteSave=0,returnState=False,
               returnEnv=False,returnEntSpec=False,
               orthonormalize=False):
    cont = True
    iterCnt = 0
    E_prev = 0
    if gaugeSiteLoad != 0:
        E,mpsL,F,EE,EEs = rightSweep(mpsL,W,F,iterCnt,
                                     nStates=nStates,
                                     alg=alg,
                                     preserveState=preserveState,
                                     startSite=gaugeSiteLoad,
                                     orthonormalize=orthonormalize)
        E,mpsL,F,EE,EEs = leftSweep(mpsL,W,F,iterCnt,
                                    nStates=nStates,
                                    alg=alg,
                                    preserveState=preserveState,
                                    orthonormalize=orthonormalize)
    while cont:
        E,mpsL,F,EE,EEs = rightSweep(mpsL,W,F,iterCnt,
                                     nStates=nStates,
                                     alg=alg,
                                     preserveState=preserveState,
                                     orthonormalize=orthonormalize)
        E,mpsL,F,EE,EEs = leftSweep(mpsL,W,F,iterCnt,
                                    nStates=nStates,
                                    alg=alg,
                                    preserveState=preserveState,
                                    orthonormalize=orthonormalize)
        cont,conv,E_prev,iterCnt = checkConv(E_prev,E,tol,iterCnt,maxIter,minIter,nStates=nStates,targetState=targetState)
    if gaugeSiteSave != 0:
        _E,mpsL,F,_EE,_EEs = rightSweep(mpsL,W,F,iterCnt+1,
                                        nStates=nStates,
                                        alg=alg,
                                        preserveState=preserveState,
                                        endSite=gaugeSiteSave,
                                        orthonormalize=orthonormalize)
        if _E is not None:
            E,EE,EEs = _E,_EE,_EEs
    save_mps(mpsL,fname,gaugeSite=gaugeSiteSave)
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
        output.append(mpsL)
    if returnEnv:
        output.append(F)
    return output

def run_dmrg(mpo,initEnv=None,initGuess=None,mbd=[2,4,8,16],
             tol=1e-5,maxIter=10,minIter=0,fname=None,
             nStates=1,targetState=0,
             constant_mbd=False,alg='arnoldi',
             preserveState=False,gaugeSiteSave=None,
             returnState=False,returnEnv=False,returnEntSpec=False,
             orthonormalize=False):
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
            mpsL = create_all_mps(N,mbdi,nStates)
            mpsL = make_all_mps_right(mpsL)
            if constant_mbd: mps = increase_mbd(mpsL,mbdi,constant=True)
            gSite = 0
        else:
            if mbdInd == 0:
                mpsL,gSite = load_mps(N,initGuess+'_mbd'+str(mbdInd),nStates=nStates)
            else:
                mpsL,gSite = load_mps(N,initGuess+'_mbd'+str(mbdInd-1),nStates=nStates)
                mpsL = increase_all_mbd(mpsL,mbdi)
        if initEnv is None: 
            env = calc_env(mpsL[0],mpo,mbdi,gaugeSite=gSite)
        else:
            env = initEnv
        fname_tmp = None
        if fname is not None: fname_tmp = fname + '_mbd' + str(mbdInd)
        output = run_sweeps(mpsL,mpo,env,
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
                              returnEntSpec=returnEntSpec,
                              orthonormalize=orthonormalize)
        # Extract Results
        E = output[0]
        EE = output[1]
        gap = output[2]
        Evec[mbdInd]  = output[0]
        EEvec[mbdInd] = output[1]
        gapvec[mbdInd]= output[2]
        if returnEntSpec and returnState and returnEnv:
            EEs = output[3]
            mpsL = output[4]
            env = output[5]
        elif returnEntSpec and returnState:
            EEs = output[3]
            mpsL = output[4]
        elif returnEntSpec and returnEnv:
            EEs = output[3]
            env = output[4]
        elif returnState and returnEnv:
            mpsL = output[3]
            env = output[4]
        elif returnEntSpec:
            EEs = output[3]
        elif returnState:
            mpsL = output[3]
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
        output.append(mpsL)
    if returnEnv:
        output.append(env)
    return output
