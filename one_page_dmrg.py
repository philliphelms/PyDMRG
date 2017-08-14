import numpy as np
import matplotlib.pyplot as plt
import time
import xlsxwriter

class simpleHeisDMRG:
    
    def __init__(self,L):
        self.wb = xlsxwriter.Workbook('progress.xlsx')
        # Input Parameters
        self.L = L
        self.d = 2
        self.D = 8
        self.h = 0
        self.J = 1
        self.tol = 1e-3
        self.max_sweep_cnt = 3
        # MPO
        S_p = np.array([[0,1],
                        [0,0]])
        S_m = np.array([[0,0],
                        [1,0]])
        S_z = np.array([[0.5,0],
                        [0,-0.5]])
        zero_mat = np.zeros([2,2])
        I = np.eye(2)
        """self.w_arr = np.array([[I,           zero_mat,      zero_mat,      zero_mat],
                               [S_z,         zero_mat,      zero_mat,      zero_mat],
                               [zero_mat,    I,             zero_mat,      zero_mat],
                               [zero_mat,    self.J*S_z,    self.J*S_z,    I       ]])
        self.w_arr = np.array([[I,           zero_mat,      zero_mat],
                               [S_z,         self.h*I,      zero_mat],
                               [zero_mat,    self.J*S_z,    I       ]])"""
        self.w_arr = np.array([[I,           zero_mat,      zero_mat,      zero_mat,   zero_mat],
                               [S_p,         zero_mat,      zero_mat,      zero_mat,   zero_mat],
                               [S_m,         zero_mat,      zero_mat,      zero_mat,   zero_mat],
                               [S_z,         zero_mat,      zero_mat,      zero_mat,   zero_mat],
                               [-self.h*S_z, self.J/2.*S_m, self.J/2.*S_p, self.J*S_z, I       ]])
        # MPS
        self.M = []
        a_prev = 1
        going_up = True
        for i in range(L):
            if going_up:
                a_curr = min(self.d**(i+1),self.d**(L-(i)))
                if a_curr <= a_prev:
                    going_up = False
                    a_curr = self.d**(L-(i+1))
                    a_prev = self.d**(L-(i))
            else:
                a_curr = self.d**(L-(i+1))
                a_prev = self.d**(L-(i))
            max_ind_curr = min([a_curr,self.D])
            max_ind_prev = min([a_prev,self.D])
            if going_up:
                newMat = np.random.rand(self.d,max_ind_curr,max_ind_prev)
            else:
                newMat = np.random.rand(self.d,max_ind_curr,max_ind_prev)
            self.M.insert(0,newMat)
            a_prev = a_curr
        for i in range(1,len(self.M))[::-1]:
            self.normalize(i,'left')
    
    def W(self,site):
        if site == 0:
            return np.expand_dims(self.w_arr[-1,:],0)
        elif site == self.L-1:
            return np.expand_dims(self.w_arr[:,0],1)
        else:
            return self.w_arr
        
    def initialize_f(self):
        self.F = []
        self.F.insert(0,np.array([[[1]]]))
        for site in range(1,self.L)[::-1]:
                self.F.insert(0,np.einsum('ijk,lmio,opq,kmq->jlp',\
                                              np.conjugate(self.M[site]),self.W(site),self.M[site],self.F[0]))
    
    def h_optimization(self,site,dir):
        #print('Using F[{}], W[{}] and F[{}]'.format(site-1,site,site))
        h = np.einsum('ijk,jlmn,olp->mionkp',self.F[site-1],self.W(site),self.F[site]) 
        si,aim,ai,sip,aimp,aip = h.shape
        h = np.reshape(h,(si*aim*ai,sip*aimp*aip))
        self.H = h
        #print(h==0)
        w,v = np.linalg.eig(h)
        w = w[(w).argsort()]
        v = v[:,(w).argsort()]
        self.M[site] = np.reshape(v[:,0],(si,aim,ai)) # could this need reshaping???
        #print('Reshaped into site {}, {}'.format(site,self.M[site].shape))
        return w[0]
    
    def normalize(self,site,dir):
        si,aim,ai = self.M[site].shape
        if dir == 'right':
            prevProd = np.dot(self.M[site],self.M[site+1])
            M_reduced = np.reshape(self.M[site],(si*aim,ai))
            (U,S,V) = np.linalg.svd(M_reduced,full_matrices=0)
            self.M[site] = np.reshape(U,(si,aim,ai))
            #print('Normalized at site {}, {}'.format(site,self.M[site].shape))
            self.M[site+1] = np.einsum('i,ij,kjl->kil',S,V,self.M[site+1])
            #print('Multiplied remainder into site {}, {}'.format(site+1,self.M[site+1].shape))
            newProd = np.dot(self.M[site],self.M[site+1])
            if np.max(np.abs(prevProd-newProd),axis=(0,1,2,3)) > 1e-3:
                raise ValueError('Normalized Sites arent identical')
            if np.sum(np.diag(np.einsum('ijk,ikl->jl',np.transpose(self.M[site],(0,2,1)),self.M[site]))) > ai+1e-3:
                if np.sum(np.diag(np.einsum('ijk,ikl->jl',np.transpose(self.M[site],(0,2,1)),self.M[site]))) < ai-1e-3:
                    raise ValueError('Normalized Sites arent producing identiy matrices')
        elif dir == 'left':
            prevProd = np.dot(self.M[site-1],self.M[site])
            M_swapped = np.swapaxes(self.M[site],0,1)
            M_reduced = np.reshape(M_swapped,(aim,si*ai))
            (U,S,V) = np.linalg.svd(M_reduced,full_matrices=0)
            self.M[site] = np.swapaxes(np.reshape(V,(aim,si,ai)),0,1)
            #print('Normalized at site {}, {}'.format(site,self.M[site].shape))
            self.M[site-1] = np.einsum('ijk,kl,l->ijl',self.M[site-1],U,S)
            #print('Multiplied remainder into site {}, {}'.format(site-1,self.M[site-1].shape))
            newProd = np.dot(self.M[site-1],self.M[site])
            if np.max(np.abs(prevProd-newProd),axis=(0,1,2,3)) > 1:
                raise ValueError('Normalized Sites arent identical')
            if np.sum(np.diag(np.einsum('ijk,ikl->jl',np.transpose(self.M[site],(0,2,1)),self.M[site]))) > ai+1e-3:
                if np.sum(np.diag(np.einsum('ijk,ikl->jl',np.transpose(self.M[site],(0,2,1)),self.M[site]))) < ai-1e-3:
                    raise ValueError('Normalized Sites arent producing identiy matrices')
        
    def update_f(self,site,dir):
        if dir == 'right':
            # Updating L expressions
            self.F[site] = np.einsum('ijkl,knm,nip,lpq->mjq',\
                                     self.W(site),np.conjugate(self.M[site]),self.F[site-1],self.M[site])
            tmpArray = np.tensordot(self.W(site),np.conjugate(self.M[site]),(2,0)) # ijlnm
            tmpArray = np.tensordot(tmpArray,self.F[site-1],([0,3],[1,0])) # jlmp
            tmpArray = np.tensordot(tmpArray,self.M[site],([1,3],[0,1])) # jmq
            #print(self.F[site] - np.swapaxes(tmpArray,0,1))
            #print('Updated L at {}, {}'.format(site,self.F[site].shape))
        elif dir == 'left':
            # Updating R expressions
            self.F[site-1] = np.einsum('ijkl,knm,mjq,lpq->nip',\
                                     self.W(site),np.conjugate(self.M[site]),self.F[site],self.M[site])
            tmpArray = np.tensordot(self.W(site),np.conjugate(self.M[site]),(2,0)) #ijlnm
            tmpArray = np.tensordot(tmpArray,self.F[site],([1,4],[1,0])) # ilnq
            tmpArray = np.tensordot(tmpArray,self.M[site],([1,3],[0,2])) # inp
            #print(self.F[site-1] - np.swapaxes(tmpArray,0,1))
            #dprint('Updated R at {}, {}'.format(site-1,self.F[site-1].shape))
        
    def print_everything(self,energy,dir,sweep_number,site):
        self.ws = self.wb.add_worksheet(dir+' sweep '+str(sweep_number)+', site '+str(site))
        # Write Energy
        self.ws.write(0,0,'Energy')
        self.ws.write(1,0,energy)
        # Write M
        self.ws.write(0,2,'M')
        si,aim,ai = self.M[site].shape
        row_cnt = 1;
        col_cnt_overall = 2
        for i in range(si):
            self.ws.write(row_cnt,col_cnt_overall,'('+str(i)+',:,:)')
            row_cnt += 1
            for j in range(aim):
                col_cnt = col_cnt_overall
                for k in range(ai):
                    self.ws.write(row_cnt,col_cnt,self.M[site][i,j,k])
                    col_cnt += 1
                row_cnt += 1
        # Write H
        col_cnt_overall = col_cnt + 1
        self.ws.write(0,col_cnt_overall,'H')
        sx,sy = self.H.shape
        row_cnt = 1
        for i in range(sx):
            col_cnt = col_cnt_overall
            for j in range(sy):
                self.ws.write(row_cnt,col_cnt,self.H[i,j])
                col_cnt += 1
            row_cnt += 1
        # Write F
        col_cnt_overall = col_cnt + 1
        self.ws.write(0,col_cnt_overall,'F')
        sx,sy,sz = self.F[site].shape
        #print(self.F[site].shape)
        row_cnt = 1
        for i in range(sx):
            self.ws.write(row_cnt,col_cnt_overall,'('+str(i)+',:,:)')
            row_cnt += 1
            for j in range(sy):
                col_cnt = col_cnt_overall
                for k in range(sz):
                    self.ws.write(row_cnt,col_cnt,self.F[site][i,j,k])
                    col_cnt += 1
                row_cnt +=1
        
    def run_optimization(self):
        converged = False
        sweep_cnt = 0
        energy_vec = [0]
        energy_vec_all = [0]
        while not converged:
            print('Beginning Sweep Set {}'.format(sweep_cnt))
            print('\tBeginning Right Sweep')
            for site in range(self.L-1):
                energy_vec_all.insert(len(energy_vec_all),self.h_optimization(site,'right'))
                self.normalize(site,'right')
                self.update_f(site,'right')
                #if energy_vec_all[-1] > energy_vec_all[-2]:
                    #print('\t\t\t!!!ENERGY INCREASE {}!!!'.format(energy_vec_all[-1]-energy_vec_all[-2]))
                print('\t\tOptimized site {}: {}'.format(site,energy_vec_all[-1]))
                #self.print_everything(energy_vec_all[-1],'right',sweep_cnt,site)
            print('\tBeginning Left Sweep')
            for site in range(1,self.L)[::-1]:
                energy_vec_all.insert(len(energy_vec_all),self.h_optimization(site,'right'))
                self.normalize(site,'left')
                self.update_f(site,'left')
                #if energy_vec_all[-1] > energy_vec_all[-2]:
                    #print('\t\t\t!!!ENERGY INCREASE {}!!!'.format(energy_vec_all[-1]-energy_vec_all[-2]))
                print('\t\tOptimized site {}: {}'.format(site,energy_vec_all[-1]))
                #self.print_everything(energy_vec_all[-1],'left',sweep_cnt,site)
            energy_vec.insert(len(energy_vec),energy_vec_all[-1])
            if np.abs(energy_vec[-1]-energy_vec[-2]) < self.tol:
                converged = True
                print(('#'*75+'\nGround state energy: {}\n'+'#'*75).format(energy_vec[-1]))
            elif sweep_cnt > self.max_sweep_cnt:
                converged = True
                print('Total number of iterations exceeded limit')
                print('Resulting calculated energy: {}'.format(energy_vec[-1]))
            else:
                sweep_cnt += 1
        self.wb.close()
    
    def calc_ground_state(self):
        self.initialize_f()
        self.run_optimization()

if __name__ == "__main__":
    t0 = time.time()
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=2)
    x = simpleHeisDMRG(100)
    x.calc_ground_state()
    t1 = time.time()
    print(('#'*75+'\nTotal Time: {}\n'+'#'*75).format(t1-t0))
