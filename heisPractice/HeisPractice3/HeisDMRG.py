import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

class HeisDMRG:
    # Class that controls the density matrix renormalization group (DMRG) method
    #
    # Functions:
    #   
    def __init__(self,mpo,mps,D,tol,max_sweep_cnt,verbose,plot_option,plot_cnt):
        self.mpo = mpo
        self.mps = mps
        self.D = D
        self.tol = tol
        self.max_sweep_cnt = max_sweep_cnt
        self.verbose = verbose
        self.plot_option = plot_option
        self.plot_cnt = plot_cnt
       
    def print_h(self,H,ntabs):
        ns,_ = H.shape
        for i in range(ns):
            print(('\t'*ntabs+'\t\t{}').format(H[i,:]))
        
    def print_m(self,site,ntabs):
        for i in range(self.mps.d):
            print(('\t'*ntabs+'\t\tOccupation {}').format(i))
            ns,_ = self.mps.M[site][i,:,:].shape
            for j in range(ns):
                print(('\t'*ntabs+'\t\t\t{}').format(self.mps.M[site][i,j,:]))
    
    def print_mmt(self,i,ntabs):
        print(('\t'*ntabs+'\tSite {}').format(i))
        result = np.dot(self.mps.M[i][0,:,:],np.transpose(self.mps.M[i][0,:,:]))+\
                 np.dot(self.mps.M[i][1,:,:],np.transpose(self.mps.M[i][1,:,:]))
        nx,_ = result.shape
        for j in range(nx):
            print(('\t'*ntabs+'\t\t{}').format(result[j,:]))
        
    def print_mtm(self,i,ntabs):
        print(('\t'*ntabs+'\tSite {}').format(i))
        result = np.dot(np.transpose(self.mps.M[i][0,:,:]),self.mps.M[i][0,:,:])+\
                 np.dot(np.transpose(self.mps.M[i][1,:,:]),self.mps.M[i][1,:,:])
        nx,_ = result.shape
        for j in range(nx):
            print(('\t'*ntabs+'\t\t{}').format(result[j,:]))
        
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
        # This function uses an eigenvalue/vector solver to determine the optimal
        # matrix M for the given site.
        #
        # Arguments:
        #   site - indicates which site we are at.
        if True:
            # My Way
            H = np.einsum('ijk,jlmn,olp->mionkp',self.mps.L_array[site],self.mpo.W(site),self.mps.R_array[site+1])
            sl,alm,al,slp,almp,alp = H.shape
            H = H.reshape(sl*alm*al,sl*alm*al) 
            if self.verbose > 3:
                print('\t'*2+'\tInitial Reshaped H for Optimization:')
                self.print_h(H,2)
            w,v = np.linalg.eig(H)
            w = np.sort(w)
            v = v[:,w.argsort()]
            if self.verbose:
                print('\t'*2+'\tInitial Matrix M:')
                self.print_m(site,2)
            # self.mps.M[site] = v[:,0].reshape(sl, alm, al)
            self.mps.M[site] = np.reshape(v[:,0],(sl,alm,al),order='C')
            if self.verbose:
                print('\t'*2+'\tCompleted Matrix M:')
                self.print_m(site,2)
            energy = w[0]
            if self.verbose > 1:
                print(('\t'*2+'\tEnergy for optimization at site {}: {}').format(site,energy))
        else:
            H = np.einsum('ijk,jlmn,olp->ikmnop',self.mps.L_array[site],self.mpo.W(site),self.mps.R_array[site+1])
            alm,almp,sl,slp,al,alp = H.shape
            H = H.reshape(sl*alm*al,sl*alm*al) # WE NEED TO EXCHANGE INDICES BEFORE DOING ThIS!!!!!!!!!
            if self.verbose > 3:
                print('\t'*2+'\tInitial Reshaped H for Optimization:')
                self.print_h(H,2)
            [w,v] = la.eigh(H)
            w = np.sort(w)
            v = v[:,w.argsort()]
            if self.verbose:
                print('\t'*2+'\tInitial Matrix M:')
                self.print_m(site,2)
            # self.mps.M[site] = v[:,0].reshape(sl, alm, al) # THIS IS THE POSSIBLE PROBLEM LINE!!!!!!!!!!!!
            self.mps.M[site] = np.reshape(v[:,0],(sl,alm,al),order='F')
            if self.verbose:
                print('\t'*2+'\tCompleted Matrix M:')
                self.print_m(site,2)
            energy = w[0]
            if self.verbose > 1:
                print(('\t'*2+'\tEnergy for optimization at site {}: {}').format(site,energy))
        return energy
    
    def normalize(self,site,direction):
        # Perform right or left normalization on M[site] using singular 
        # value documentation, then multiply the remaining matrices into 
        # the next site in the sweep.
        si,aim,ai = self.mps.M[site].shape
        trySwapping = True
        if trySwapping: # This is possibly a fix to a problem, but I don't know yet........
            M_swapped_inds = np.swapaxes(self.mps.M[site],0,1)
        if direction == 'right':
            M_2d = self.mps.M[site].reshape(aim*si,ai)
            (U,S,V) = np.linalg.svd(M_2d,full_matrices=0)
            if trySwapping:
                U_3d = U.reshape(aim,si,ai)
                U_3d = np.swapaxes(U_3d,0,1)
                self.mps.M[site] = U_3d
            else:
                self.mps.M[site] = U.reshape(si,aim,ai)    
            self.mps.M[site+1] = np.einsum('i,ij,kjl->kil',S,V,self.mps.M[site+1])
            if self.verbose:
                print('\t'*2+'\tNormalized Matrix M:')
                self.print_m(site,2)
                print('\t'*2+'\tNew Matrix at site + 1:')
                self.print_m(site+1,2)
                if self.verbose > 1:
                    print('\t'*2+'\tProof of Normalization:')
                    self.print_mtm(site,2)
        elif direction == 'left':
            M_2d = self.mps.M[site].reshape(aim,si*ai)
            (U,S,V) = np.linalg.svd(M_2d,full_matrices=0)
            if trySwapping:
                V_3d = V.reshape(aim,si,ai)
                V_3d = np.swapaxes(V_3d,0,1)
                self.mps.M[site] = V_3d
            else:
                self.mps.M[site] = V.reshape(si,aim,ai)
            self.mps.M[site-1] = np.einsum('ijk,kl,l->ijl',self.mps.M[site-1],U,S)
            if self.verbose:
                print('\t'*2+'\tNormalized Matrix M:')
                self.print_m(site,2)
                print('\t'*2+'\tNew Matrix at site - 1:')
                self.print_m(site-1,2)
                if self.verbose > 1:
                    print('\t'*2+'\tProof of Normalization:')
                    self.print_mmt(site,2)
        else:
            raise ValueError('Sweep Direction must be left or right')
        
    
    def run_optimization(self):
        converged = False
        calculate_energy_separately = False
        sweep_cnt = 0
        energy_prev = 0
        self.update_plot(0,True)
        while not converged:
            if self.verbose:
                print('#'*68)
            print('Beginning Sweep Set {}'.format(sweep_cnt))
            print('\tRight Sweep')
            for site in range(self.mps.L-1):
                if self.verbose:
                    print('\t\t'+'-'*52)
                print('\t\tOptimizing Site {}'.format(site))
                energy_curr = self.H_opt(site)
                self.normalize(site,'right')
                self.mps.update_lr(site,'right',self.mpo.W)
                if calculate_energy_separately:
                    energy_curr = self.mps.calc_energy(site,self.mpo.W)
                self.update_plot(energy_curr,False)
            print('\tLeft Sweep')
            for site in range(self.mps.L-1,0,-1):
                if self.verbose:
                    print('\t\t'+'-'*52)
                print('\t\tOptimizing Site {}'.format(site))
                energy_curr = self.H_opt(site)
                self.normalize(site,'left')
                self.mps.update_lr(site,'left',self.mpo.W)
                print(site)
                if calculate_energy_separately:
                    energy_curr = self.mps.calc_energy(site,self.mpo.W)
                if site == 1:
                    self.update_plot(energy_curr,True)
                else:
                    self.update_plot(energy_curr,False)
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