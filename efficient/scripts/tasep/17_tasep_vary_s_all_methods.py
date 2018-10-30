import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# We calculate the CGF and Current for the TASEP at many values of s, then 
# Compare the resulting profiles for DMRG, Mean Field, and Exact Diagonalization
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

# Run TASEP Current Calculations
N_vec = np.array([6,8])
s_vec = np.linspace(-1,1,20)
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
fig1.savefig('varyS_CGF_comparison.pdf')
fig2.savefig('varyS_scaledCGF_comparison.pdf')
fig3.savefig('varyS_scaledCurrent_comparison.pdf')
fig4.savefig('varyS_current_comparison.pdf')
fig5.savefig('varyS_comparison_CGFerror.pdf')
fig6.savefig('varyS_comparison_currentError.pdf')
