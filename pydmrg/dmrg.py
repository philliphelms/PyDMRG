import numpy as np
import scipy.linalg as sla
from pyscf.lib import eig as davidson
from pyscf.lib import einsum
from scipy.sparse.linalg import eigs as arnoldi
from scipy.sparse.linalg import LinearOperator
from tools.mps_tools import *
from tools.mpo_tools import *
from tools.diag_tools import *
from tools.env_tools import *
from tools.contract import *
import warnings

VERBOSE = 10

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
    if VERBOSE > 2: print('\t\tEE = {}'.format(EE))
    return EE, EEs

def calc_ent_left(M,v,site):
    (n1,n2,n3) = M[site].shape
    Mtmp = np.reshape(v,(n1,n2,n3))
    M_reshape = np.swapaxes(Mtmp,0,1)
    M_reshape = np.reshape(M_reshape,(n2,n1*n3))
    (_,S,_) = np.linalg.svd(M_reshape,full_matrices=False)
    EE,EEs = calc_entanglement(S)
    if VERBOSE > 2: print('\t\tEE = {}'.format(EE))
    return EE, EEs

def renormalizeR(mpsL,v,site,nStates=1,targetState=0):
    (n1,n2,n3) = mpsL[0][site].shape
    # Try to calculate Entanglement?
    EE,EEs = calc_ent_right(mpsL[0],v[:,targetState],site)
    # Calculate the reduced density matrix
    _,nStatesCalc = v.shape
    nStatesAvg = min(nStates,nStatesCalc)
    for i in range(nStatesAvg):
        if nStatesAvg != 1: 
            vtmp = v[:,i]
        else:
            vtmp  = v
        vReshape = np.reshape(vtmp,(n1,n2,n3))
        w = 1./float(nStatesAvg)
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
            if VERBOSE > 4: print('\t\tNormalization Problem')
        # Calculate next site for guess
        if nStates != 1:
            vReshape = np.reshape(v[:,min(nStatesAvg-1,state)],(n1,n2,n3))
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
    nStatesAvg = min(nStates,nStatesCalc)
    for i in range(nStatesAvg):
        if nStatesAvg != 1: 
            vtmp = v[:,i]
        else: 
            vtmp = v
        vReshape = np.reshape(vtmp,(n1,n2,n3))
        w = 1./float(nStatesAvg)
        if i == 0:
            rdm = w*calcRDM(vReshape,'left') 
        else:
            rdm +=w*calcRDM(vReshape,'left')
    # Take eigenvalues of the rdm
    vals,vecs = np.linalg.eig(rdm) 
    # Sort inds
    inds = np.argsort(vals)[::-1]
    # Keep only maxBondDim eigenstates
    inds = inds[:n2]
    vals = vals[inds]
    vecs = vecs[:,inds]
    # Make sure vecs are orthonormal
    vecs = sla.orth(vecs)
    vecs = vecs.T
    # Loops through all MPSs in list
    for state in range(nStates):
        # Put resulting vectors into MPS
        mpsL[state][site] = np.reshape(vecs,(n2,n1,n3))
        mpsL[state][site] = np.swapaxes(mpsL[state][site],0,1)
        if not np.all(np.isclose(einsum('ijk,ilk->jl',mpsL[state][site],np.conj(mpsL[state][site])),np.eye(n2),atol=1e-6)):
            if VERBOSE > 4: print('\t\tNormalization Problem')
        # Calculate next site's guess
        if nStates != 1:
            vReshape = np.reshape(v[:,min(nStatesAvg-1,state)],(n1,n2,n3))
        else:
            vReshape = np.reshape(v,(n1,n2,n3))
        # Push gauge onto next site
        mpsL[state][site-1] = einsum('ijk,lkm,lnm->ijn',mpsL[state][site-1],vReshape,np.conj(mpsL[state][site]))
    return mpsL,EE,EEs

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
    if VERBOSE > 1: print('Right Sweep {}'.format(iterCnt))
    for site in range(startSite,endSite):
        E,mpsL,F,_EE,_EEs = rightStep(mpsL,W,F,site,
                                      nStates,
                                      alg=alg,
                                      preserveState=preserveState,
                                      orthonormalize=orthonormalize)
        if VERBOSE > 2: print('\tEnergy at Site {}: {}'.format(site,E))
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
    if VERBOSE > 1: print('Left Sweep {}'.format(iterCnt))
    for site in range(startSite,endSite,-1):
        E,mpsL,F,_EE,_EEs = leftStep(mpsL,W,F,site,
                                     nStates,
                                     alg=alg,
                                     preserveState=preserveState,
                                     orthonormalize=orthonormalize)
        if VERBOSE > 2: print('\tEnergy at Site {}: {}'.format(site,E))
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

def printResults(converged,E,EE,EEspec,gap):
    if VERBOSE > 1: print('#'*75)
    if converged:
        if VERBOSE > 0: print('Converged at E = {}'.format(E))
    else:
        if VERBOSE > 0: print('Convergence not acheived, E = {}'.format(E))
    if VERBOSE > 2: print('\tGap = {}'.format(gap))
    if VERBOSE > 2: print('\tEntanglement Entropy  = {}'.format(EE))
    if VERBOSE > 3: print('\tEntanglement Spectrum =')
    for i in range(len(EEspec)):
        if VERBOSE > 3: print('\t\t{}'.format(EEspec[i]))
    if VERBOSE > 1: print('#'*75)

def run_sweeps(mpsL,W,F,initGuess=None,maxIter=0,minIter=None,
               tol=1e-10,fname = None,nStates=1,
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
        # Do final calculation 
        _,v,_ = calc_eigs(mpsL,W,F,gaugeSiteSave-1,
                         nStates,
                         alg=alg,
                         preserveState=preserveState,
                         orthonormalize=orthonormalize)
        # Put final result into mpsL
        (n1,n2,n3) = mpsL[0][gaugeSiteSave-1].shape
        for state in range(nStates):
            mpsL[state][gaugeSiteSave-1] = np.reshape(v[:,state],(n1,n2,n3))
        # Check if we got to the center site
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
             orthonormalize=False,calcLeftState=False):
    # Determine number of sites from length of mpo operators
    N = len(mpo[0])

    # Set to save MPS at center site as default
    if gaugeSiteSave is None: gaugeSiteSave = int(N/2)+1
    
    # Check Data Structures to make sure they are correct
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

    # Get mpo for calculating left state
    if calcLeftState: mpol = mpo_conj_trans(mpo)

    # Create data structures to save results
    Evec  = np.zeros(len(mbd),dtype=np.complex_)
    EEvec = np.zeros(len(mbd),dtype=np.complex_)
    gapvec= np.zeros(len(mbd),dtype=np.complex_)
    if calcLeftState: EEvecl = np.zeros(len(mbd),dtype=np.complex_)

    # Loop over all maximum bond dimensions, running dmrg for each one
    for mbdInd,mbdi in enumerate(mbd):
        if VERBOSE > 1: print('Starting Calc for MBD = {}'.format(mbdi))

        # Set up initial MPS
        if initGuess is None:
            
            # Make random or constant MPS initial guess
            mpsList = create_all_mps(N,mbdi,nStates)
            mpsList = make_all_mps_right(mpsList)
            # PH - constant_mbd not currently working
            if constant_mbd: mps = increase_mbd(mpsList,mbdi,constant=True)
            # Right canonical, so set gauge site at 0
            gSite = 0

            # Repeat for left eigenstate
            if calcLeftState:
                mpslList = create_all_mps(N,mbdi,nStates)
                mpslList = make_all_mps_right(mpsList)
                if constant_mbd: mps = increase_mbd(mpslList,mbdi,constant=True)
                glSite = 0
        else: # PH - Should check if it is a sting here and add the additional possibility that the input is an mpsList

            # Load user provided MPS guess    
            if mbdInd == 0:
                # Load user provided MPS Guess
                mpsList,gSite = load_mps(initGuess+'_mbd'+str(mbdInd))
                # Repeat for left eigenstate
                if calcLeftState: mpslList,glSite = load_mps(initGuess+'_mbd'+str(mbdInd)+'_left')
            else:
                # Load mps guess from previous bond dimension and increase to current mbd
                mpsList,gSite = load_mps(initGuess+'_mbd'+str(mbdInd-1))
                mpsList = increase_all_mbd(mpsList,mbdi)
                # Repeat for left eigenstate
                if calcLeftState:
                    mpslList,glSite = load_mps(initGuess+'_mbd'+str(mbdInd-1)+'_left')
                    mpslList = increase_all_mbd(mpslList,mbdi)

        # Calc environment (or load if provided)
        if initEnv is None: 
            env = calc_env(mpsList[0],mpo,mbdi,gaugeSite=gSite)
            if calcLeftState: envl = calc_env(mpslList[0],mpol,mbdi,gaugeSite=glSite)
        else:
            env = initEnv
            if calcLeftState: env,envl = initEnv[0],initEnv[1]

        # Add an index to the MPS filename saving to indicate its bond dimension
        fname_mbd = None
        if fname is not None: fname_mbd = fname + '_mbd' + str(mbdInd)

        # Run DMRG Sweeps (right eigenvector)
        if VERBOSE > 0: print('Calculating Right Eigenstate')
        output = run_sweeps(mpsList,mpo,env,
                              maxIter=maxIter[mbdInd],
                              minIter=minIter[mbdInd],
                              tol=tol[mbdInd],
                              fname=fname_mbd,
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
        # Extract Extra results
        if returnEntSpec and returnState and returnEnv:
            EEs = output[3]
            mpsList = output[4]
            env = output[5]
        elif returnEntSpec and returnState:
            EEs = output[3]
            mpsList = output[4]
        elif returnEntSpec and returnEnv:
            EEs = output[3]
            env = output[4]
        elif returnState and returnEnv:
            mpsList = output[3]
            env = output[4]
        elif returnEntSpec:
            EEs = output[3]
        elif returnState:
            mpsList = output[3]
        elif returnEnv:
            env = output[3]

        if calcLeftState:
            # Run DMRG Sweeps (left eigenvector)
            if VERBOSE > 0: print('Calculating Left Eigenstate')
            output = run_sweeps(mpslList,mpol,envl,
                                  maxIter=maxIter[mbdInd],
                                  minIter=minIter[mbdInd],
                                  tol=tol[mbdInd],
                                  fname=fname_mbd+'_left',
                                  nStates=nStates,
                                  alg=alg,
                                  targetState=targetState,
                                  gaugeSiteLoad=glSite,
                                  gaugeSiteSave=gaugeSiteSave,
                                  preserveState=preserveState,
                                  returnState=returnState,
                                  returnEnv=returnEnv,
                                  returnEntSpec=returnEntSpec,
                                  orthonormalize=orthonormalize)
            # Extract left state specific Results
            EEl = output[1]
            EEvecl[mbdInd]  = output[1]
            # Extra potential extra data
            if returnEntSpec and returnState and returnEnv:
                EEsl = output[3]
                mpslList = output[4]
                envl = output[5]
            elif returnEntSpec and returnState:
                EEsl = output[3]
                mpslList = output[4]
            elif returnEntSpec and returnEnv:
                EEsl = output[3]
                envl = output[4]
            elif returnState and returnEnv:
                mpslList = output[3]
                envl = output[4]
            elif returnEntSpec:
                EEsl = output[3]
            elif returnState:
                mpslList = output[3]
            elif returnEnv:
                envl = output[3]

    # Lump right and left results
    if calcLeftState:
        EE = [EE,EEl]
        if returnEntSpec: EEs = [EEs,EEsl]
        if returnState: mpsList = [mpsList,mpslList]
        if returnEnv: env = [env,envl]

    # Return Results
    if len(Evec) == 1:
        output = [E,EE,gap]
    else:
        output = [Evec,EEvec,gapvec]
    if returnEntSpec:
        output.append(EEs)
    if returnState:
        output.append(mpsList)
    if returnEnv:
        output.append(env)
    return output
