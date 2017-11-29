import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

# Possible calculations:
simple_tasep = False
vary_systemSize = False
vary_s = False
vary_maxBondDim = False
phaseDiagram = True
simpleHeis = False
simpleFullSEP = False
reverseFullSEP = False
heis2D = False
simpleIsing = False
check_2d_tasep = False

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

if simple_tasep:
    # Run single TASEP calculation
    x = mps_opt.MPS_OPT(N = 10,
                        hamType = 'tasep',
                        plotExpVals = False,
                        plotConv = False,
                        hamParams = (0.35,-1,2/3))
    x.kernel()

if vary_systemSize:
    N_vec = np.linspace(2,100,50)
    s = np.array([-0.01,0.01])
    current = np.zeros_like(N_vec)
    for i in range(len(N_vec)):
        N = N_vec[i]
        print('Running Calcs for N={}'.format(N))
        x = mps_opt.MPS_OPT(N=N,maxIter=5,hamType='tasep',hamParams=(3/5,s[0],2/3))
        E_left = x.kernel()
        x = mps_opt.MPS_OPT(N=N,maxIter=5,hamType='tasep',hamParams=(3/5,s[1],2/3))
        E_right = x.kernel()
        current[i] = (E_right-E_left)/(s[1]-s[0])/(N+1)
    fig1 = plt.figure()
    plt.plot(N_vec,-current,'ro-',markersize=5,linewidth=3)
    plt.xlabel('$N$',fontsize=20)
    plt.ylabel('$J(N)/(N+1)$',fontsize=20)
    fig1.savefig('VarySize.pdf')

if vary_s:
    # Run TASEP Current Calculations
    N_vec = np.array([10,20,30,40,50,60])
    s_vec = np.linspace(-1,1,100)
    fig1 = plt.figure(10)
    fig2 = plt.figure(20)
    fig3 = plt.figure(30)
    fig4 = plt.figure(40)
    col_vec = ['r','r','y','g','b','c','k','m']
    for j in range(len(N_vec)):
        N = N_vec[j]
        print('Running Calcs for N={}'.format(N))
        Evec = np.zeros_like(s_vec)
        Evec_adj = np.zeros_like(s_vec)
        for i in range(len(s_vec)):
            print('\tRunning Calcs for s={}'.format(s_vec[i]))
            x = mps_opt.MPS_OPT(N =int(N),
                                  maxIter = 5,
                                  hamType = "tasep",
                                  hamParams = (0.35,s_vec[i],2/3))
            Evec[i] = x.kernel()
            Evec_adj[i] = Evec[i]/(N+1)
        Ediff = Evec[1:]-Evec[:len(Evec)-1]
        Sdiff = s_vec[1:]-s_vec[:len(s_vec)-1]
        slope = -Ediff/(Sdiff)
        plt.figure(10)
        plt.plot(s_vec,Evec,col_vec[j]+'-',linewidth=3)
        plt.hold(True)
        plt.xlabel('$s$',fontsize=20)
        plt.ylabel('$\mu$',fontsize=20)
        plt.figure(20)
        plt.plot(s_vec,Evec_adj,col_vec[j]+'-',linewidth=3)
        plt.hold(True)
        plt.xlabel('$s$',fontsize=20)
        plt.ylabel('$\mu/(N+1)$',fontsize=20)
        plt.figure(30)
        plt.plot(s_vec[1:],slope,col_vec[j]+'-',linewidth=3)
        plt.xlabel('$s$',fontsize=20)
        plt.ylabel('$\partial_s\mu=J$',fontsize=20)
        plt.hold(True)
        plt.figure(40)
        plt.plot(s_vec[1:],slope/(N+1),col_vec[j]+'-',linewidth=3)
        plt.xlabel('$s$',fontsize=20)
        plt.ylabel('$\partial_s\mu/(N+1)$',fontsize=20)
    fig1.savefig('VaryS_CGF.pdf')
    fig2.savefig('varyS_scaledCGF.pdf')
    fig3.savefig('VaryS_scaledCurrent.pdf')
    fig4.savefig('VaryS_current.pdf')

if vary_maxBondDim:
    N = 10
    bondDimVec = np.array([2])
    E_exact = 5.378053311010889458998462941963
    Evec = np.zeros(len(bondDimVec))
    diffVec = np.zeros(len(bondDimVec))
    for i in range(len(bondDimVec)):
        print('\tRunning Calcs for M = {}'.format(bondDimVec[i]))
        x = mps_opt.MPS_OPT(N=int(N),
                            maxBondDim = bondDimVec[i],
                            tol = 1e-1,
                            hamParams = (0.35,-1,2/3))
        Evec[i] = x.kernel()
    #diffVec = np.abs(Evec-E_exact)
    diffVec = np.abs(Evec-Evec[-1])
    fig = plt.figure()
    print(diffVec)
    plt.semilogy(bondDimVec,diffVec,'b-',linewidth=5)
    plt.hold(True)
    plt.semilogy(bondDimVec,diffVec,'ro',markersize=10)
    #plt.plot(np.array([bondDimVec[0],bondDimVec[-1]]),np.array([0,0]),'b--',linewidth=5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Bond Dimension',fontsize=20)
    plt.ylabel('$E-E_{exact}$',fontsize=20)
    fig.savefig('varyMaxBondDim.pdf')

if phaseDiagram:
    N = 10
    npts = 100
    betaVec = np.linspace(0,1,npts)
    alphaVec = np.linspace(0,1,npts)
    J_mat = np.zeros((len(betaVec),len(alphaVec)))
    J_mat_inf = np.zeros((len(betaVec),len(alphaVec)))
    for i in range(len(betaVec)):
        for j in range(len(alphaVec)):
            print('-'*20+'\nalpha = {}%, beta = {}%\n'.format(j/len(alphaVec),i/len(betaVec)))
            #x = mps_opt.MPS_OPT(N=int(N),
                                maxBondDim = 8,
                                tol = 1e-1,
                                hamParams = (alphaVec[j],-0.001,betaVec[i]))
            E1 = x.kernel()
            x = mps_opt.MPS_OPT(N=int(N),
                                maxBondDim = 8,
                                tol = 1e-1,
                                hamParams = (alphaVec[j],0.001,betaVec[i]))
            E2 = x.kernel()
            J_mat[i,j] = (E1-E2)/(0.002)/N
            # Determine infinite limit current
            if alphaVec[j] > 0.5 and betaVec[i] > 0.5:
                J_mat_inf[i,j] = 1/4
            elif alphaVec[j] < 0.5 and betaVec[i] > alphaVec[j]:
                J_mat_inf[i,j] = alphaVec[j]*(1-alphaVec[j])
            else:
                J_mat_inf[i,j] = betaVec[i]*(1-betaVec[i])
    x,y = np.meshgrid(betaVec,alphaVec)
    f = plt.figure()
    plt.pcolor(x,y,J_mat,vmin=-0,vmax=0.25)
    plt.colorbar()
    plt.hold(True)
    plt.plot(np.array([0,0.5]),np.array([0,0.5]),'k-',linewidth=5)
    plt.plot(np.array([0.5,0.5]),np.array([0.5,1]),'k-',linewidth=5)
    plt.plot(np.array([0.5,1]),np.array([0.5,0.5]),'k-',linewidth=5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('$\alpha$',fontsize=20)
    plt.ylabel('$\beta$',fontsize=20)
    f.savefig('my_dmrg_phaseDiagram.pdf')
    f2 = plt.figure()
    plt.pcolor(x,y,J_mat_inf,vmin=-0,vmax=0.25)
    plt.colorbar()
    plt.hold(True)
    plt.plot(np.array([0,0.5]),np.array([0,0.5]),'k-',linewidth=5)
    plt.plot(np.array([0.5,0.5]),np.array([0.5,1]),'k-',linewidth=5)
    plt.plot(np.array([0.5,1]),np.array([0.5,0.5]),'k-',linewidth=5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('a',fontsize=20)
    plt.ylabel('b',fontsize=20)
    f2.savefig('my_analytic_phaseDiagram.pdf')

if simpleHeis:
    N = 50
    x = mps_opt.MPS_OPT(N=int(N),
                        hamType = "heis",
                        plotExpVals = True,
                        plotConv = True,
                        hamParams = (1,0))
    E = x.kernel()

if simpleFullSEP:
    N = 8
    x = mps_opt.MPS_OPT(N=N,
                        hamType = "sep",
                        plotExpVals = True,
                        plotConv = True,
                        hamParams = (0.35,0,1,0,0,2/3,-1))
    E = x.kernel()

if reverseFullSEP:
    N = 8
    x = mps_opt.MPS_OPT(N=N,
                        hamType = "sep",
                        plotExpVals = True,
                        plotConv = True,
                        hamParams = (0,2/3,0,1,0.35,0,-1))
    E = x.kernel()

if heis2D:
    N = 10
    x = mps_opt.MPS_OPT(N=N**2,
                        hamType = "heis_2d",
                        plotExpVals = True,
                        plotConv = True,
                        maxBondDim=4,
                        hamParams = (1,0))
    E = x.kernel()

if simpleIsing:
    N = 50
    x = mps_opt.MPS_OPT(N=N,
                        hamType = "ising",
                        plotExpVals = True,
                        plotConv = True,
                        hamParams = (1,0))
    E = x.kernel()

if check_2d_tasep:
    # This checks that the 2D sep calculation gives the same results as the tasep
    # for all four possible directions
    N = 10
    x = mps_opt.MPS_OPT(N=N**2,
                        hamType="sep_2d",
                        plotExpVals=True,
                        plotConv=True,
                        hamParams = (0,1,0.35,0,0,2/3,      # jl,jr,il,ir,ol,or,
                                     0,0,0,   0,0,0  ,-1))  # ju,jd,it,ib,ot,ob,s
    E = x.kernel()
