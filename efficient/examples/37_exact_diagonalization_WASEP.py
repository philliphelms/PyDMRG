import numpy as np
import time
import mps_opt
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Ensure that the 2D sep calculations are correct by doing a tasep calculation
# aligned in all four possible directions. Compare the results to ensure the 
# resulting energies are coincident.
#-----------------------------------------------------------------------------

# Set Plotting parameters
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
np.set_printoptions(suppress=True)
np.set_printoptions(precision=100)
plt.style.use('ggplot') #'fivethirtyeight') #'ggplot'

Nx = 2
Ny = Nx
n_points = 10
E = 10
px = 1/2*np.exp(-E/Nx)
qx = 1/2*np.exp(E/Nx)
s = np.linspace(-20,20,20)
s = np.array([-10])
jl = qx*np.ones((Ny,Nx))
jr = px*np.ones((Ny,Nx))
ju = 1/2*np.ones((Ny,Nx))
jd = 1/2*np.ones((Ny,Nx))
cr = np.zeros((Ny,Nx))
cl = np.zeros((Ny,Nx))
cd = np.zeros((Ny,Nx))
cu = np.zeros((Ny,Nx))
dr = np.zeros((Ny,Nx))
dl = np.zeros((Ny,Nx))
dd = np.zeros((Ny,Nx))
du = np.zeros((Ny,Nx))



CGF = np.zeros(s.shape)
for i in range(len(s)):
    print('s={}'.format(s[i]))
    x1 = mps_opt.MPS_OPT(N=[Nx,Ny],
                        hamType="sep_2d",
                        plotExpVals=True,
                        plotConv=True,
                        maxBondDim = 10,#,300],#,500],
                        tol = 1e-3,#,1e-3],#,1e-4],
                        maxIter=4,#,3],#,10],
                        max_eig_iter=50,
                        verbose=3,
                        periodic_x=True,
                        periodic_y=True,
                        add_noise = False,
                        hamParams = (jl,jr,jd,ju,cr,cl,cd,cu,dr,dl,dd,du,[-s[i]/Nx,0]))
                        #hamParams = (qx,px,0,0,0,0,0.5,0.5,0,0,0,0,[s[i]/Nx,0]))
                        #(jump_left,jump_right,enter_left,enter_right,
                        # exit_left,exit_right,jump_up,jump_down,
                        # enter_top,enter_bottom,exit_top,exit_bottom,s))
    x1.initialize_containers()
    x1.generate_mps()
    x1.generate_mpo()
    Hamiltonian = x1.mpo.return_full_ham()
    print(Hamiltonian)
    #CGF[i] = x1.kernel()

plt.plot(s,CGF)
plt.show()
