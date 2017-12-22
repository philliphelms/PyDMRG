import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

## Possible calculations:#########################
simple_tasep = False
vary_systemSize = False
vary_s = False
vary_maxBondDim = False
phaseDiagram = False
simpleHeis = False
simpleFullSEP = False
reverseFullSEP = False
heis2D = False
simpleIsing = False
check_2d_tasep = False
practice_2d_tasep = False
test_ds = False
# Comparing DMRG, MF & ED
vary_s_ed = True
vary_s_comp = False
vary_maxBondDim_comp = False
phaseDiagram_comp = False
# Full 2D Comparison
vary_maxBondDim_2d_comp = False
##################################################


# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

if simple_tasep:
    # Run single TASEP calculation
    x = mps_opt.MPS_OPT(N = 12,
                        hamType = 'tasep',
                        plotExpVals = True,
                        plotConv = True,
                        hamParams = (0.35,-1,2/3))
    x.kernel()

if vary_systemSize:
    N_vec = np.array([10,20,30,40,50,60,70,80,90,100,150,200,250,300,400,500,600,700,800,900,1000])
    s = np.array([-0.01,0.01])
    current = np.zeros(len(N_vec))
    for i in range(len(N_vec)):
        N = N_vec[i]
        print('Running Calcs for N={}'.format(N))
        x = mps_opt.MPS_OPT(N=N,hamType='tasep',hamParams=(3/5,s[0],2/3))
        E_left = x.kernel()
        x = mps_opt.MPS_OPT(N=N,hamType='tasep',hamParams=(3/5,s[1],2/3))
        E_right = x.kernel()
        current[i] = (E_right-E_left)/(s[1]-s[0])/(N+1)
    fig1 = plt.figure()
    plt.plot(N_vec,-current,'ro-',markersize=5,linewidth=3)
    plt.xlabel('$N$',fontsize=20)
    plt.ylabel('$J(N)/(N+1)$',fontsize=20)
    fig1.savefig('VarySize.pdf')

if vary_s:
    # Run TASEP Current Calculations
    N_vec = np.array([4])#np.array([10,20,30,40,50,60])
    s_vec = np.linspace(-10,10,100)
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
        plt.xlabel('$s$',fontsize=20)
        plt.ylabel('$\mu$',fontsize=20)
        plt.figure(20)
        plt.plot(s_vec,Evec_adj,col_vec[j]+'-',linewidth=3)
        plt.xlabel('$s$',fontsize=20)
        plt.ylabel('$\mu/(N+1)$',fontsize=20)
        plt.figure(30)
        plt.plot(s_vec[1:],slope,col_vec[j]+'-',linewidth=3)
        plt.xlabel('$s$',fontsize=20)
        plt.ylabel('$\partial_s\mu=J$',fontsize=20)
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
            x = mps_opt.MPS_OPT(N=int(N),
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
    N = 4
    x1 = mps_opt.MPS_OPT(N=N**2,
                        hamType="sep_2d",
                        plotExpVals=True,
                        plotConv=True,
                        hamParams = (0,1,0.35,0,0,2/3,      # jl,jr,il,ir,ol,or,
                                     0,0,0,   0,0,0  ,-1))  # ju,jd,it,ib,ot,ob,s
    E1 = x1.kernel()
    x2 = mps_opt.MPS_OPT(N=N**2,
                         hamType="sep_2d",
                         plotExpVals=True,
                         plotConv=True,
                         hamParams = (1,0,0,0.35,2/3,0,
                                      0,0,0,0   ,0  ,0,-1))
    E2 = x2.kernel()
    x3 = mps_opt.MPS_OPT(N=N**2,
                         hamType="sep_2d",
                         plotExpVals=True,
                         plotConv=True,
                         hamParams = (0,0,0,0,0,0,
                                      1,0,0,0.35,2/3,0,-1))
    E3 = x3.kernel()
    x4 = mps_opt.MPS_OPT(N=N**2,
                         hamType="sep_2d",
                         plotExpVals=True,
                         plotConv=True,
                         hamParams = (0,0,0,0,0,0,
                                      0,1,0.35,0,0,2/3,-1))
    E4 = x4.kernel()

if practice_2d_tasep:
    N = 10
    x = mps_opt.MPS_OPT(N=N**2,
                        hamType="sep_2d",
                        plotExpVals=True,
                        plotConv=True,
                        hamParams = (0.5,0.5,0.9,0.2,0.2,0.8,
                                     0.5,0.5,0.9,0.2,0.2,0.8,0))
    E = x.kernel()

if test_ds:
    # Find the optimal spacing for ds
    N = 6
    npts = 5
    ds = np.array([0.5,0.4,0.3,0.2,0.1,0.05,0.01,0.001,0.0001])
    error = np.zeros(ds.shape)
    betaVec = np.linspace(0.01,0.99,npts)
    alphaVec = np.linspace(0.01,0.99,npts)
    J_mat = np.zeros((len(betaVec),len(alphaVec)))
    J_mat_inf = np.zeros((len(betaVec),len(alphaVec)))
    J_mat_ed = np.zeros((len(betaVec),len(alphaVec)))
    J_mat_mf = np.zeros((len(betaVec),len(alphaVec)))
    for k in range(len(ds)):
        for i in range(len(betaVec)):
            for j in range(len(alphaVec)):
                print('-'*20+'\nalpha = {}% Complete\nbeta = {} Complete%\n'.format(j/len(alphaVec)*100,i/len(betaVec)*100))
                x = mps_opt.MPS_OPT(N=int(N),
                                    hamParams = (alphaVec[j],-ds[k],betaVec[i]))
                E1 = x.kernel()
                print(E1)
                E1_ed = x.exact_diag()
                print(E1_ed)
                print(E1-E1_ed)
                x = mps_opt.MPS_OPT(N=int(N),
                                    hamParams = (alphaVec[j],ds[k],betaVec[i]))
                E2 = x.kernel()
                print(E2)
                E2_ed = x.exact_diag()
                print(E2_ed)
                print(E2-E2_ed)
                # Calculate Current
                J_mat[i,j] = (E1-E2)/(2*ds[k])/N
                J_mat_ed[i,j] = (E1_ed-E2_ed)/(2*ds[k])/N
        error[k] = np.sum(np.sum(np.abs(J_mat-J_mat_ed)))/(len(alphaVec)*len(betaVec))
    plt.figure()
    plt.semilogy(ds,np.abs(error))
    plt.show()

if vary_s_ed:
    # Recreate Ushnish plot
    N = 8
    s_vec = np.linspace(-2,2,20)
    E = np.zeros(s_vec.shape)
    for i in range(len(s_vec)):
        x = mps_opt.MPS_OPT(N=N,
                            hamType = "sep",
                            hamParams = (0.9,0.1,0.5,0.5,0.9,0.1,s_vec[i]),
                            usePyscf = False)
        x.kernel()
        E[i] = x.exact_diag()
    plt.plot(s_vec,E)
    plt.grid(True)
    plt.show()

if vary_s_comp:
    # Run TASEP Current Calculations
    N_vec = np.array([10])#np.array([10,20,30,40,50,60])
    s_vec = np.linspace(-1,1,100)
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()
    fig5 = plt.figure()
    fig6 = plt.figure()
    col_vec = ['r','r','y','g','b','c','k','m']
    for j in range(len(N_vec)):
        N = N_vec[j]
        print('Running Calcs for N={}'.format(N))
        Evec = np.zeros_like(s_vec)
        Evec_ed = np.zeros_like(s_vec)
        Evec_mf = np.zeros_like(s_vec)
        Evec_adj = np.zeros_like(s_vec)
        Evec_ed_adj = np.zeros_like(s_vec)
        Evec_mf_adj = np.zeros_like(s_vec)

        for i in range(len(s_vec)):
            print('\tRunning Calcs for s={}'.format(s_vec[i]))
            # Run DMRG
            x = mps_opt.MPS_OPT(N =int(N),
                                  maxIter = 5,
                                  hamType = "tasep",
                                  hamParams = (0.35,s_vec[i],2/3))
            Evec[i] = x.kernel()
            Evec_adj[i] = Evec[i]/(N+1)
            # Run Exact Diagonalization
            Evec_ed[i] = x.exact_diag()
            Evec_ed_adj[i] = Evec_ed[i]/(N+1)
            # Run Mean Field
            Evec_mf[i] = x.mean_field()
            Evec_mf_adj[i] = Evec_mf[i]/(N+1)
        # Calculate Slopes
        Ediff = Evec[1:]-Evec[:len(Evec)-1]
        E_ed_diff = Evec_ed[1:]-Evec_ed[:len(Evec_ed)-1]
        E_mf_diff = Evec_mf[1:]-Evec_mf[:len(Evec_ed)-1]
        Sdiff = s_vec[1:]-s_vec[:len(s_vec)-1]
        slope = -Ediff/(Sdiff)
        slope_ed = -E_ed_diff/(Sdiff)
        slope_mf = -E_mf_diff/(Sdiff)
        # Plot CGF vs. s
        plt.figure(fig1.number)
        plt.plot(s_vec,Evec,col_vec[j]+'-',linewidth=2)
        plt.plot(s_vec,Evec_ed,col_vec[j]+':',linewidth=2)
        plt.plot(s_vec,Evec_mf,col_vec[j]+'--',linewidth=2)
        plt.xlabel('$s$',fontsize=20)
        plt.ylabel('$\mu$',fontsize=20)
        plt.legend(('DMRG','Exact Diagonalization','Mean Field'))
        # Plot Scaled CGF vs. s
        plt.figure(fig2.number)
        plt.plot(s_vec,Evec_adj,col_vec[j]+'-',linewidth=2)
        plt.plot(s_vec,Evec_ed_adj,col_vec[j]+':',linewidth=2)
        plt.plot(s_vec,Evec_mf_adj,col_vec[j]+'--',linewidth=2)
        plt.xlabel('$s$',fontsize=20)
        plt.ylabel('$\mu/(N+1)$',fontsize=20)
        plt.legend(('DMRG','Exact Diagonalization','Mean Field'))
        # Plot Current vs. s
        plt.figure(fig3.number)
        plt.plot(s_vec[1:],slope,col_vec[j]+'-',linewidth=3)
        plt.plot(s_vec[1:],slope_ed,col_vec[j]+':',linewidth=2)
        plt.plot(s_vec[1:],slope_mf,col_vec[j]+'--',linewidth=2)
        plt.xlabel('$s$',fontsize=20)
        plt.ylabel('$\partial_s\mu=J$',fontsize=20)
        plt.legend(('DMRG','Exact Diagonalization','Mean Field'))
        # Plot Scaled Current vs. s
        plt.figure(fig4.number)
        plt.plot(s_vec[1:],slope/(N+1),col_vec[j]+'-',linewidth=3)
        plt.plot(s_vec[1:],slope_ed/(N+1),col_vec[j]+':',linewidth=2)
        plt.plot(s_vec[1:],slope_mf/(N+1),col_vec[j]+'--',linewidth=2)
        plt.xlabel('$s$',fontsize=20)
        plt.ylabel('$\partial_s\mu/(N+1)$',fontsize=20)
        plt.legend(('DMRG','Exact Diagonalization','Mean Field'))
        # Plot % Error for CGF
        plt.figure(fig5.number)
        plt.semilogy(s_vec,1e-16+np.abs(Evec-Evec_ed)/Evec_ed*100,col_vec[j]+'-',linewidth=2)
        plt.semilogy(s_vec,1e-16+np.abs(Evec_mf-Evec_ed)/Evec_ed*100,col_vec[j]+':',linewidth=2)
        plt.xlabel('$s$',fontsize=20)
        plt.ylabel('Percent Error (CGF)',fontsize=20)
        plt.legend(('DMRG','Mean Field'))
        # Plot % Error for Current
        plt.figure(fig6.number)
        plt.semilogy(s_vec[1:],1e-16+np.abs(slope-slope_ed)/slope_ed*100,col_vec[j]+'-',linewidth=2)
        plt.semilogy(s_vec[1:],1e-16+np.abs(slope_mf-slope_ed)/slope_ed*100,col_vec[j]+':',linewidth=2)
        plt.xlabel('$s$',fontsize=20)
        plt.ylabel('Percent Error (Current)',fontsize=20)
        plt.legend(('DMRG','Mean Field'))
    fig1.savefig('VaryS_CGF_comparison.pdf')
    fig2.savefig('varyS_scaledCGF_comparison.pdf')
    fig3.savefig('VaryS_scaledCurrent_comparison.pdf')
    fig4.savefig('VaryS_current_comparison.pdf')
    fig5.savefig('VaryS_comparison_CGFerror.pdf')
    fig6.savefig('VaryS_comparison_currentError.pdf')

if vary_maxBondDim_comp:
    N = np.array([4,6,8,10])
    bondDimVec = np.array([1,2,3,4,10])
    col_vec = ['r','y','g','b','c','k','m']
    fig1 = plt.figure()
    for j in range(len(N)):
        Evec = np.zeros(len(bondDimVec))
        diffVec = np.zeros(len(bondDimVec))
        for i in range(len(bondDimVec)):
            print('\tRunning Calcs for M = {}'.format(bondDimVec[i]))
            x = mps_opt.MPS_OPT(N=int(N[j]),
                                maxBondDim = bondDimVec[i],
                                tol = 1e-1,
                                hamParams = (0.35,-1,2/3))
            Evec[i] = x.kernel()
        #E_ed = x.exact_diag()
        E_mf = x.mean_field()
        #diffVec = np.abs(Evec-E_exact)
        diffVec = np.abs(Evec-Evec[-1])
        plt.semilogy(bondDimVec[:-1],diffVec[:-1],col_vec[j]+'-o',linewidth=5,markersize=10,markeredgecolor='k')
        plt.semilogy([bondDimVec[0],bondDimVec[-2]],[np.abs(E_mf-Evec[-1]),np.abs(E_mf-Evec[-1])],col_vec[j]+':',linewidth=5)
        #plt.plot(np.array([bondDimVec[0],bondDimVec[-1]]),np.array([0,0]),'b--',linewidth=5)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
    plt.xlabel('Bond Dimension',fontsize=20)
    plt.ylabel('$E-E_{exact}$',fontsize=20)
    plt.legend(('DMRG','Mean Field'))
    plt.show()
    fig1.savefig('varyMaxBondDim.pdf')

if phaseDiagram_comp:
    N = 10
    npts = 50
    ds = 0.1
    betaVec = np.linspace(0.01,0.99,npts)
    alphaVec = np.linspace(0.01,0.99,npts)
    J_mat = np.zeros((len(betaVec),len(alphaVec)))
    J_mat_inf = np.zeros((len(betaVec),len(alphaVec)))
    J_mat_ed = np.zeros((len(betaVec),len(alphaVec)))
    J_mat_mf = np.zeros((len(betaVec),len(alphaVec)))
    for i in range(len(betaVec)):
        for j in range(len(alphaVec)):
            print(('-'*20+'\nalpha = {}\nbeta = {}\n{}% Complete\n'+'-'*20).format(alphaVec[j],betaVec[i],i/len(betaVec)*100))
            x = mps_opt.MPS_OPT(N=int(N),
                                hamParams = (alphaVec[j],-ds,betaVec[i]))
            E1 = x.kernel()
            E1_ed = x.exact_diag()
            E1_mf = x.mean_field()
            x = mps_opt.MPS_OPT(N=int(N),
                                hamParams = (alphaVec[j],ds,betaVec[i]))
            E2 = x.kernel()
            E2_ed = x.exact_diag()
            E2_mf = x.mean_field()
            # Calculate Current
            J_mat[i,j] = (E1-E2)/(2*ds)/N
            J_mat_ed[i,j] = (E1_ed-E2_ed)/(2*ds)/N
            J_mat_mf[i,j] = (E1_mf-E2_mf)/(2*ds)/N
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
    plt.plot(np.array([0,0.5]),np.array([0,0.5]),'k-',linewidth=5)
    plt.plot(np.array([0.5,0.5]),np.array([0.5,1]),'k-',linewidth=5)
    plt.plot(np.array([0.5,1]),np.array([0.5,0.5]),'k-',linewidth=5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('$\alpha$',fontsize=20)
    plt.ylabel('$\beta$',fontsize=20)
    f.savefig('dmrg_phaseDiagram.pdf')
    f2 = plt.figure()
    plt.pcolor(x,y,J_mat_inf,vmin=-0,vmax=0.25)
    plt.colorbar()
    plt.plot(np.array([0,0.5]),np.array([0,0.5]),'k-',linewidth=5)
    plt.plot(np.array([0.5,0.5]),np.array([0.5,1]),'k-',linewidth=5)
    plt.plot(np.array([0.5,1]),np.array([0.5,0.5]),'k-',linewidth=5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('a',fontsize=20)
    plt.ylabel('b',fontsize=20)
    f2.savefig('analytic_phaseDiagram.pdf')
    f3 = plt.figure()
    plt.pcolor(x,y,J_mat_ed,vmin=-0,vmax=0.25)
    plt.colorbar()
    plt.plot(np.array([0,0.5]),np.array([0,0.5]),'k-',linewidth=5)
    plt.plot(np.array([0.5,0.5]),np.array([0.5,1]),'k-',linewidth=5)
    plt.plot(np.array([0.5,1]),np.array([0.5,0.5]),'k-',linewidth=5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('$\alpha$',fontsize=20)
    plt.ylabel('$\beta$',fontsize=20)
    f3.savefig('ed_phaseDiagram.pdf')
    f4 = plt.figure()
    plt.pcolor(x,y,J_mat_mf,vmin=-0,vmax=0.25)
    plt.colorbar()
    plt.plot(np.array([0,0.5]),np.array([0,0.5]),'k-',linewidth=5)
    plt.plot(np.array([0.5,0.5]),np.array([0.5,1]),'k-',linewidth=5)
    plt.plot(np.array([0.5,1]),np.array([0.5,0.5]),'k-',linewidth=5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('$\alpha$',fontsize=20)
    plt.ylabel('$\beta$',fontsize=20)
    f4.savefig('mf_phaseDiagram.pdf')
    f5 = plt.figure()
    plt.pcolor(x,y,np.log(np.abs(J_mat_ed-J_mat+1e-16)))
    plt.colorbar()
    plt.plot(np.array([0,0.5]),np.array([0,0.5]),'k-',linewidth=5)
    plt.plot(np.array([0.5,0.5]),np.array([0.5,1]),'k-',linewidth=5)
    plt.plot(np.array([0.5,1]),np.array([0.5,0.5]),'k-',linewidth=5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('$\alpha$',fontsize=20)
    plt.ylabel('$\beta$',fontsize=20)
    f5.savefig('dmrg_phaseDiagram_error.pdf')
    f6 = plt.figure()
    plt.pcolor(x,y,np.log(np.abs(J_mat_ed-J_mat_mf+1e-16)))
    plt.colorbar()
    plt.plot(np.array([0,0.5]),np.array([0,0.5]),'k-',linewidth=5)
    plt.plot(np.array([0.5,0.5]),np.array([0.5,1]),'k-',linewidth=5)
    plt.plot(np.array([0.5,1]),np.array([0.5,0.5]),'k-',linewidth=5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('$\alpha$',fontsize=20)
    plt.ylabel('$\beta$',fontsize=20)
    f6.savefig('mf_phaseDiagram_error.pdf')

if vary_maxBondDim_2d_comp:
    N = 12
    bondDimVec = np.array([20])
    col_vec = ['r','y','g','b','c','k','m']
    # Run 1D Calculation for comparison
    Evec_1d = np.zeros(len(bondDimVec))
    diffVec = np.zeros(len(bondDimVec))
    print('Running 1D Calculations')
    for i in range(len(bondDimVec)):
        x = mps_opt.MPS_OPT(N=N,
                            maxBondDim = bondDimVec[i],
                            hamParams = (0.35,-1,2/3),
                            plotConv = False,
                            plotExpVals = False,
                            hamType = 'tasep')
        Evec_1d[i] = x.kernel()
    # Run exact Diagonalization for 1D
    print('Running Exact Diagonalization (1D)')
    E_ed = x.exact_diag()
    # Run mean field 1d
    print('Running mean field (1D)')
    E_mf = x.mean_field()
    # Run 2D in opposite direction
    Evec_2d_notaligned = np.zeros(len(bondDimVec))
    print('Running misaligned 2D calculations')
    for i in range(len(bondDimVec)):
        x = mps_opt.MPS_OPT(N=N**2,
                            maxBondDim = bondDimVec[i],
                            hamType="sep_2d",
                            plotExpVals=False,
                            plotConv=False,
                            verbose = 3,
                            hamParams = (0,0,0,0,0,0,
                                         1,0,0,0.35,2/3,0,-1))
        Evec_2d_notaligned[i] = x.kernel()/N
        plt.close("all")
    # Run 2D in aligned direction
    Evec_2d_aligned = np.zeros(len(bondDimVec))
    print('Running aligned 2D calculations')
    for i in range(len(bondDimVec)):
        x = mps_opt.MPS_OPT(N=N**2,
                            maxBondDim = bondDimVec[i],
                            hamType="sep_2d",
                            plotExpVals=False,
                            plotConv=False,
                            hamParams = (0,1,0.35,0,0,2/3,      # jl,jr,il,ir,ol,or,
                                         0,0,0,   0,0,0  ,-1))  # ju,jd,it,ib,ot,ob,s
        Evec_2d_aligned[i] = x.kernel()/N
    # Calculate Errors
    err_mf = np.abs(E_mf-E_ed)
    errVec_1d = np.abs(Evec_1d-E_ed)
    errVec_2d_aligned = np.abs(Evec_2d_aligned-E_ed)
    errVec_2d_notaligned = np.abs(Evec_2d_notaligned-E_ed)
    # Create Plot
    fig1 = plt.figure()
    plt.semilogy(np.array([np.min(bondDimVec),np.max(bondDimVec)]),np.array([err_mf,err_mf]),':',linewidth=3)
    plt.semilogy(bondDimVec,errVec_1d,linewidth=3)
    plt.semilogy(bondDimVec,errVec_2d_aligned,linewidth=3)
    plt.semilogy(bondDimVec,errVec_2d_notaligned,linewidth=3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Bond Dimension',fontsize=20)
    plt.ylabel('$E-E_{exact}$',fontsize=20)
    plt.legend(('Mean Field','1D DMRG','2D DMRG (aligned)','2D DMRG (not aligned)'))
    fig1.savefig('varyMaxBondDim_'+str(bondDimVec[i])+'.pdf')
