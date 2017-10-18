import numpy as np
import time
import mps_dmrg
import matplotlib.pyplot as plt

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)
plt.style.use('ggplot')
if False:
    # Run Heisenberg Calculation
    x = mps_dmrg.MPS_DMRG(L = 50,
                          ham_type = 'heis',
                          ham_params = (1,-1))
    x.calc_ground_state()

if True:
    # Run single TASEP calculation
    x = mps_dmrg.MPS_DMRG(L = 4,
                          ham_type = "tasep",
                          ham_params = (0.35,-1,2/3))
    x.calc_ground_state()

if False:
    # Calculate 
    alpha = 0.35
    beta = 2/3
    x = mps_dmrg.MPS_DMRG(L=30,
                          max_sweep_cnt = 3,
                          ham_type = "tasep",
                          fileName = ('profileResults.npz'),
                          ham_params = (alpha,0,beta),
                          verbose = 10)
    x.calc_ground_state()
    plt.figure(10)
    plt.plot(range(0,x.L,2),x.calc_full[0::2],'r:',linewidth=5)
    plt.xlabel('Site',fontsize=20)
    plt.ylabel('Expected Occupancy',fontsize=20)
    frame1 = plt.gca()
    plt.setp(frame1.get_xticklabels(), fontsize=14)
    plt.setp(frame1.get_yticklabels(), fontsize=14)
    plt.show()

if False:
    # Calculate Average Occupancies for range of alpha and beta values
    alpha_vec = np.array([0.2,0.8])
    beta_vec = np.array([0.2,0.8])
    for i in range(len(alpha_vec)):
        for j in range(len(beta_vec)):
            x = mps_dmrg.MPS_DMRG(L=20,
                                  max_sweep_cnt = 20,
                                  ham_type = "tasep",
                                  ham_params = (alpha_vec[i],0,beta_vec[j]),
                                  verbose = 10,
                                  plotConv = True,
                                  plotExpVal = True)
            x.calc_ground_state()
            plt.figure(10)
            #plt.plot(range(1,x.L),x.calc_full[1:],':')
            plt.plot(range(0,x.L,2),x.calc_full[0::2],'r:',linewidth=5)
            plt.xlabel('Site',fontsize=20)
            plt.ylabel('Expected Occupancy',fontsize=20)
            frame1=plt.gca()
            plt.setp(frame1.get_xticklabels(), fontsize=14)
            plt.setp(frame1.get_yticklabels(), fontsize=14)
    plt.show()

if False:
    # Run TASEP Current Calculations
    N_vec = np.array([6])
    s_vec = np.linspace(-1,1,20)
    #s_vec = np.array([-1,-0.8,-0.6,-0.4,-0.2,-0.1,-0.05,-0.005,0,0.005,0.05,0.1,0.2,0.4,0.6,0.8,1.0])
    plt.figure(10)
    plt.figure(20)
    col_vec = ['r','r','y','g','b','c']
    for j in range(len(N_vec)):
        N = N_vec[j]
        print('Running Calcs for N={}'.format(N))
        Evec = np.zeros_like(s_vec)
        Evec_adj = np.zeros_like(s_vec)
        Evec_adj2 = np.zeros_like(s_vec)
        for i in range(len(s_vec)):
            print('\tRunning Calcs for s={}'.format(s_vec[i]))
            t0 = time.time()
            np.set_printoptions(suppress=True)
            np.set_printoptions(precision=2)
            x = mps_dmrg.MPS_DMRG(L = N,
                                  max_sweep_cnt = 3,
                                  ham_type = "tasep",
                                  fileName = ('data/Results_'+str(N)+'_'+str(i)+'.npz'),
                                  ham_params = (3/5,s_vec[i],2/3))
            Evec[i] = x.calc_ground_state()
            Evec_adj[i] = Evec[i]/(N+1)
            t1 = time.time()
        Ediff = Evec[1:]-Evec[:len(Evec)-1]
        Sdiff = s_vec[1:]-s_vec[:len(s_vec)-1]
        slope = -Ediff/(Sdiff*(N+1))
        plt.figure(10)
        plt.plot(s_vec,Evec,col_vec[j]+'o-')
        plt.hold(True)
        plt.xlabel('$s$',fontsize=20)
        plt.ylabel('$\mu$',fontsize=20)
        plt.figure(20)
        plt.plot(s_vec,Evec_adj,col_vec[j]+'o-')
        plt.hold(True)
        plt.xlabel('$s$',fontsize=20)
        plt.ylabel('$\mu/(N+1)$',fontsize=20)
        plt.figure(30)
        plt.plot(s_vec[1:],slope,col_vec[j]+'o-')
        plt.xlabel('$s$',fontsize=20)
        plt.ylabel('$\partial_s\mu/(N+1)$',fontsize=20)
        plt.hold(True)
    plt.show()
