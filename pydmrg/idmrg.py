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

def return_bulk_mpo(mpo):
    mpoBulk = []
    nOps = len(mpo)
    for opInd in range(nOps):
        op = [None]*2
        op[0] = mpo[opInd][1]
        op[1] = mpo[opInd][2]
        mpoBulk.append(op)
    return mpoBulk

def return_edge_mpo(mpo):
    mpoEdge = []
    nOps = len(mpo)
    for opInd in range(nOps):
        op = [None]*2
        op[0] = mpo[opInd][0]
        op[1] = mpo[opInd][3]
        mpoEdge.append(op)
    return mpoEdge

def make_next_guess(mps,mbd=10,d=2):
    nStates = len(mps)
    for state in range(nStates):
        (n1,n2,n3) = mps[state][0].shape
        mps[state][0] = np.pad(mps[state][0],((0,0),(0,min(mbd,n3)-n2),(0,min(mbd,n3*d)-n3)),'constant')
        mps[state][1] = np.pad(mps[state][1],((0,0),(0,min(mbd,n3*d)-n3),(0,min(mbd,n3)-n2)),'constant')
    return mps

def single_iter(N,mps,mpo,env,nStates,alg='davidson',mbd=10):
    # Solve Eigenproblem
    E,vecs,_ = calc_eigs(mps,mpo,env,0,nStates,
                         alg=alg,oneSite=False,edgePreserveState=False)
    # Do SVD
    mps,EE,EEs,S = renorm_inf(mps,mpo,env,vecs,mbd)
    # Update Environments
    env = update_env_inf(mps[0],mpo,env,mpsl=None)
    # Update MPS (increase bond dim)
    mps = make_next_guess(mps,mbd=mbd)
    return N,E,EE,EEs,mps,env,S

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

def checkConv(E_prev,E,tol,iterCnt,maxIter=100,minIter=None,nStates=1,targetState=0):
    if nStates != 1: E = E[targetState]
    if (np.abs(E-E_prev) < tol) and (iterCnt > minIter):
        cont = False
        conv = True
    elif iterCnt > maxIter - 1:
        cont = False
        conv = False
    else:
        iterCnt += 1
        E_prev = E
        cont = True
        conv = False
    return cont,conv,E_prev,iterCnt

def run_iters(mps,mpo,env,mbd=10,maxIter=100,minIter=None,
               tol=1e-10,fname=None,nStates=1,
               targetState=0,alg='davidson',
               preserveState=False,returnState=False,
               returnEnv=False,returnEntSpec=False,
               orthonormalize=False):
    # Get Edge and bulk mpos separated
    # PH - How to handle this is initial environment is already loaded...
    mpoEdge = return_edge_mpo(mpo)
    mpoBulk = return_bulk_mpo(mpo)

    # Do first step (using edge mpo)
    N = 2
    if True: # PH - Find way to determine if we have a new environment
        N,E,EE,EEs,mps,env,S = single_iter(2,mps,mpoEdge,env,nStates,alg=alg,mbd=mbd)

    # Run iterations
    cont = True
    iterCnt = 0
    E_prev = E
    N += 2
    while cont:
        # Run single update
        N,E,EE,EEs,mps,env,S = single_iter(N,mps,mpoBulk,env,nStates,alg=alg,mbd=mbd)
        # Print Results
        print('N={}\tEnergy = {:f}\tdiff={:f}\tEE={:f}'.format(N,np.real(E/N),np.real(np.abs(E/N-E_prev)),EE))
        # Check for convergence
        cont,conv,E_prev,iterCnt = checkConv(E_prev,E/N,tol,iterCnt,maxIter,minIter,nStates=nStates,targetState=targetState)
        N += 2

    # Save Resulting MPS
    save_mps(mps,fname,gaugeSite=0)

    # Determine which vars to return
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

def run_idmrg(mpo,initEnv=None,initGuess=None,mbd=[2,4,8,16],
             tol=1e-2,maxIter=1000,minIter=10,fname=None,
             nStates=1,targetState=0,alg='davidson',
             preserveState=False,edgePreserveState=False,
             returnState=False,returnEnv=False,
             returnEntSpec=False,orthonormalize=False,
             calcLeftState=False):

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
            # Generate Random Guess
            mps = create_all_mps(2,mbdi,nStates)
            if calcLeftState: mpsl = create_all_mps(2,mbdi,nStates)
        else: 
            # Load user provided Guess    
            if mbdInd == 0:
                mps,_ = load_mps(initGuess+'_mbd'+str(mbdInd))
                if calcLeftState: mpsl,_ = load_mps(initGuess+'_mbd'+str(mbdInd)+'_left')

        # Calc initial environment (or load if provided)
        if initEnv is None: 
            env = calc_env_inf(mps,mpo,mbdi)
            if calcLeftState: envl = calc_env_inf(mpsl,mpol,mbdi)
        else:
            env = initEnv
            if calcLeftState: env,envl = initEnv[0],initEnv[1]

        # Add an index to the MPS filename saving to indicate its bond dimension
        fname_mbd = None
        if fname is not None: fname_mbd = fname + '_mbd' + str(mbdInd)

        # Run DMRG Sweeps (right eigenvector)
        if VERBOSE > 0: print('Calculating Right Eigenstate')
        output = run_iters(mps,mpo,env,
                          mbd=mbdi,
                          maxIter=maxIter[mbdInd],
                          minIter=minIter[mbdInd],
                          tol=tol[mbdInd],
                          fname=fname_mbd,
                          nStates=nStates,
                          alg=alg,
                          targetState=targetState,
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
            output = run_iters(mpslList,mpol,envl,
                              mbd=mbdi,
                              maxIter=maxIter[mbdInd],
                              minIter=minIter[mbdInd],
                              tol=tol[mbdInd],
                              fname=fname_mbd+'_left',
                              nStates=nStates,
                              alg=alg,
                              targetState=targetState,
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

if __name__ == "__main__":
    from mpo.asep import return_mpo
    # Hamiltonian Parameters
    p = 0.1 
    alpha = 0.2      # in at left
    gamma = 1.-alpha  # Out at left
    q     = 1.-p      # Jump left
    beta  = 0.4     # Out at right
    delta = 1.-beta   # In at right
    s = -0.5
    # Get MPO
    hamParams = np.array([alpha,gamma,p,q,beta,delta,s])
    mpo = return_mpo(4,hamParams)
    # Run idmrg
    output = run_idmrg(mpo,mbd=10)
