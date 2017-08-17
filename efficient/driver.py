import numpy as np
import time
import mps_dmrg
import matplotlib.pyplot as plt

if False:
    # Settings
    N = 14
    d = 2
    D = 8
    tol = 1e-3
    max_sweep_cnt = 20
    ham_type = "heis"
    ham_params = (-1,0)
    # Run Ground State Calculations
    t0 = time.time()
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=2)
    x = mps_dmrg.MPS_DMRG(L = N,
                          d = d,
                          D = D,
                          tol = tol,
                          max_sweep_cnt = max_sweep_cnt,
                          ham_type = ham_type,
                          ham_params = ham_params)
    x.calc_ground_state()
    t1 = time.time()
    print(('#'*75+'\nTotal Time: {}\n'+'#'*75).format(t1-t0))

if True:
    N_vec = np.array([100])
    s_vec = np.array([0])
    #s_vec = np.array([-1,-0.8,-0.6,-0.4,-0.2,-0.1,-0.05,-0.005,0,0.005,0.05,0.1,0.2,0.4,0.6,0.8,1.0])
    plt.figure(1)
    plt.figure(2)
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
                                  d = 2,
                                  D = 8,
                                  tol = 1e-4,
                                  max_sweep_cnt = 100,
                                  plot = False,
                                  ham_type = "tasep",
                                  verbose = 3,
                                  fileName = ('data/Results_'+str(N)+'_'+str(i)+'.npz'),
                                  ham_params = (1,s_vec[i],0))
            Evec[i] = x.calc_ground_state()
            Evec_adj[i] = Evec[i]/(N+1)
            t1 = time.time()
        Ediff = Evec[1:]-Evec[:len(Evec)-1]
        Sdiff = s_vec[1:]-s_vec[:len(s_vec)-1]
        slope = -Ediff/(Sdiff*(N+1))
        plt.figure(1)
        plt.plot(s_vec,Evec,'o:')
        plt.hold(True)
        plt.figure(2)
        plt.plot(s_vec,Evec_adj,'o:')
        plt.hold(True)
        plt.figure(3)
        plt.plot(s_vec[1:],slope,'o:')
    plt.show()



if False:
    N_vec = np.array([10,20,30,40,50,60])
    gap = 0.01
    s_vec = np.array([-gap/2,gap/2])
    plt.figure()
    slope = np.zeros(len(N_vec))
    for j in range(len(N_vec)):
        N = N_vec[j]
        Evec = np.zeros_like(s_vec)
        t0 = time.time()
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=2)
        x = mps_dmrg.MPS_DMRG(L = N,
                              d = 2,
                              D = 8,
                              tol = 1e-3,
                              max_sweep_cnt = 100,
                              plot = False,
                              ham_type = "tasep",
                              ham_params = (0.35,s_vec[0],2/3))
        Evec[0] = x.calc_ground_state()
        x = mps_dmrg.MPS_DMRG(L = N,
                              d = 2,
                              D = 8,
                              tol = 1e-3,
                              max_sweep_cnt = 100,
                              plot = False,
                              ham_type = "tasep",
                              ham_params = (0.35,s_vec[1],2/3))
        Evec[1] = x.calc_ground_state()
        slope[j] = (Evec[1]-Evec[0])/gap
        t1 = time.time()
        print('({} - {})/{} = {}'.format(Evec[1],Evec[0],gap,(Evec[1]-Evec[0])/gap))
        print(slope[j])
        print(('#'*75+'\nTotal Time: {}\n'+'#'*75).format(t1-t0))
    plt.plot(N_vec,slope)
    plt.show()
