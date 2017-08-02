import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

class HeisDMRG:
    """
    Description:
        An object containing all of the information and functions to run the DMRG
        algorithm to calculate the ground state of the 1D Heisenberg Model for a
        chain of length L.
        
    Class Members:
        > self.mpo             - The heisenberg model matrix product operator object
        > self.mps             - The heisenberg model matrix product state object
        > self.tol             - Tolerance for energy convergence criteria
        > self.max_sweep_cnt   - The maximum number of sweeps to be performed before
                                 cancelling the calculation
        > self.reshape_order   - The ordering for reshaping of matrices, should always
                                 be set at "F", indicating Fortran ordering.
        > self.plot_option     - If true, then generate matplotlib plots showing convergence
        > self.plot_cnt        - Counts the number of times the plot's been updated
    
    Key Functions:
        1) H_opt(site)         - A function that forms and solves the eigenvalue problem
                                 associated with minimizing the systems energy at the given
                                 site, then places the resulting eigenvector back into the 
                                 MPS. Done according to Equations 34-36 of the accompanying
                                 theoretical description.
        2) run_optimization()  - A simple driver that carries out each sweep and runs the 
                                 functions associated with the optimization and normalization
                                 of the MPS at each step of the algorithm.
        3) update_plot()       - Updates the convergence plot.
    """
    def __init__(self,mpo,mps,tol,max_sweep_cnt,reshape_order,plot_option):
        self.mpo = mpo
        self.mps = mps
        self.tol = tol
        self.max_sweep_cnt = max_sweep_cnt
        self.reshape_order = reshape_order
        self.plot_option = plot_option
        self.plot_cnt = 0
        
    def update_plot(self,new_entry,end_sweep):
        # Create a plot that shows the energy as a function of the number 
        # of optimizations performed
        if self.plot_option:
            if self.plot_cnt == 0:
                plt.style.use('ggplot')
                plt.ion()
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(111)
                self.conv_xdat = [self.plot_cnt]
                self.conv_ydat = [new_entry]
                if end_sweep:
                    self.conv_xdat_end_swp = [self.plot_cnt]
                    self.conv_ydat_end_swp = [new_entry]
                self.ax.plot(self.conv_xdat,self.conv_ydat,'b-')
                self.ax.hold(True)
                self.ax.plot(self.conv_xdat_end_swp,self.conv_ydat_end_swp,'ro',linestyle=':',linewidth=3)
                self.ax.hold(False)
                plt.xlabel('Iteration')
                plt.ylabel('Energy')
                self.fig.canvas.draw()
            else:
                self.conv_xdat.insert(len(self.conv_xdat),self.plot_cnt)
                self.conv_ydat.insert(len(self.conv_ydat),new_entry)
                if end_sweep:
                    self.conv_xdat_end_swp.insert(len(self.conv_xdat_end_swp),self.plot_cnt)
                    self.conv_ydat_end_swp.insert(len(self.conv_xdat_end_swp),new_entry)
                self.ax.plot(self.conv_xdat,self.conv_ydat,'b-')
                self.ax.hold(True)
                self.ax.plot(self.conv_xdat_end_swp,self.conv_ydat_end_swp,'ro',linestyle=':',linewidth=3)
                self.ax.hold(False)
                self.fig.canvas.draw()
            self.plot_cnt += 1
            
    def H_opt(self,site):
        H = np.einsum('ijk,jlmn,olp->mionkp',self.mps.L_array[site],self.mpo.W(site),self.mps.R_array[site+1])
        sl,alm,al,slp,almp,alp = H.shape
        H = np.reshape(H,(sl*alm*al,sl*alm*al),order=self.reshape_order)
        w,v = np.linalg.eig(H)
        w = np.sort(w)
        v = v[:,w.argsort()]
        self.mps.M[site] = np.reshape(v[:,0],(sl,alm,al),order=self.reshape_order)
        energy = w[0]
        return energy
    
    def run_optimization(self):
        converged = False
        sweep_cnt = 0
        energy_prev = 0
        self.update_plot(0,True)
        while not converged:
            print('Beginning Sweep Set {}'.format(sweep_cnt))
            print('\tRight Sweep')
            for site in range(self.mps.L-1):
                energy_curr = self.H_opt(site)
                self.mps.normalize(site,'right')
                self.mps.update_lr(site,'right',self.mpo.W)
                print('\t\tCompleted Site {}: {}'.format(site,energy_curr))
                if self.plot_option:
                    self.update_plot(energy_curr,False)
            print('\tLeft Sweep')
            for site in range(self.mps.L-1,0,-1):
                energy_curr = self.H_opt(site)
                self.mps.normalize(site,'left')
                self.mps.update_lr(site,'left',self.mpo.W)
                print('\t\tCompleted Site {}: {}'.format(site,energy_curr))
                if self.plot_option:
                    if site == 1:
                        self.update_plot(energy_curr,True)
                    else:
                        self.update_plot(energy_curr,False)
            # Check for convergence
            if np.abs(energy_prev-energy_curr) < self.tol:
                converged = True
                print('#'*68)
                print('System has converged at:')
                print('E = {}'.format(energy_curr))
                print('#'*68)
            elif sweep_cnt >= self.max_sweep_cnt-1:
                converged = True
                print('Maximum number of sweeps exceeded - system not converged')
            else:
                print('\tResulting Energy: {}'.format(energy_curr))
                energy_prev = energy_curr
                sweep_cnt += 1