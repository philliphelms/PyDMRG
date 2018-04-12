import mps_opt
import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Run a check to ensure PBC are working, by setting up staggered
# SEPs on a 2D lattice in both directions.
#-----------------------------------------------------------------------------

# Run 1D for comparison
N = 4
a = 2/3
b = 0.35
s = -1
dividing_point = 3
x = mps_opt.MPS_OPT(N=N,
                    hamType = "sep",
                    plotExpVals = True,
                    hamParams = (a,0,1,0,0,b,s))
E = x.kernel()
print('Expected Energy Result = {}'.format(E*N))
# Run 2D in x-direction
Nx = N
Ny = N
jl = np.zeros((Nx,Ny))
jr = np.ones((Nx,Ny))
for i in range(Ny):
    jr[i,i-1] = 0
ju = np.zeros((Nx,Ny))
jd = np.zeros((Nx,Ny))
cr = np.zeros((Nx,Ny))
for i in range(Ny):
    cr[i,i] = a
cl = np.zeros((Nx,Ny))
cu = np.zeros((Nx,Ny))
cd = np.zeros((Nx,Ny))
dr = np.zeros((Nx,Ny))
for i in range(Ny):
    dr[i,i-1] = b
dl = np.zeros((Nx,Ny))
du = np.zeros((Nx,Ny))
dd = np.zeros((Nx,Ny))
x = mps_opt.MPS_OPT(N=[Nx,Ny],
                    hamType = 'sep_2d',
                    periodic_x = True,
                    periodic_y = True,
                    plotExpVals = True,
                    plotConv = True,
                    hamParams = (jl,jr,jd,ju,cr,cl,cd,cu,dr,dl,dd,du,[s,0]))
E = x.kernel()
# Run 2D in reverse x-direction
Nx = N
Ny = N
jl = np.ones((Nx,Ny))
for i in range(Ny):
    jl[i,i] = 0
jr = np.zeros((Nx,Ny))
ju = np.zeros((Nx,Ny))
jd = np.zeros((Nx,Ny))
cr = np.zeros((Nx,Ny))
cl = np.zeros((Nx,Ny))
for i in range(Ny):
    cl[i,i-1] = a
cu = np.zeros((Nx,Ny))
cd = np.zeros((Nx,Ny))
dr = np.zeros((Nx,Ny))
dl = np.zeros((Nx,Ny))
for i in range(Ny):
    dl[i,i] = b
du = np.zeros((Nx,Ny))
dd = np.zeros((Nx,Ny))
x = mps_opt.MPS_OPT(N=[Nx,Ny],
                    hamType = 'sep_2d',
                    periodic_x = True,
                    periodic_y = True,
                    plotExpVals = True,
                    plotConv = True,
                    hamParams = (jl,jr,jd,ju,cr,cl,cd,cu,dr,dl,dd,du,[-s,0]))
E = x.kernel()
# Run 2D in y-direction
Nx = N
Ny = N
jl = np.zeros((Nx,Ny))
jr = np.zeros((Nx,Ny))
ju = np.ones((Nx,Ny))
for i in range(Nx):
    ju[i,i] = 0
jd = np.zeros((Nx,Ny))
cr = np.zeros((Nx,Ny))
cl = np.zeros((Nx,Ny))
cu = np.zeros((Nx,Ny))
for i in range(Nx):
    cu[i-1,i] = a
cd = np.zeros((Nx,Ny))
dr = np.zeros((Nx,Ny))
dl = np.zeros((Nx,Ny))
du = np.zeros((Nx,Ny))
for i in range(Nx):
    du[i,i] = b
dd = np.zeros((Nx,Ny))
x = mps_opt.MPS_OPT(N=[Nx,Ny],
                    hamType = 'sep_2d',
                    periodic_x = True,
                    periodic_y = True,
                    plotExpVals = True,
                    plotConv = True,
                    hamParams = (jl,jr,jd,ju,cr,cl,cd,cu,dr,dl,dd,du,[0,s]))
E = x.kernel()
# Run 2D in reverse y-direction
Nx = N
Ny = N
jl = np.zeros((Nx,Ny))
jr = np.zeros((Nx,Ny))
ju = np.zeros((Nx,Ny))
jd = np.ones((Nx,Ny))
for i in range(Nx):
    jd[i-1,i] = 0
cr = np.zeros((Nx,Ny))
cl = np.zeros((Nx,Ny))
cu = np.zeros((Nx,Ny))
cd = np.zeros((Nx,Ny))
for i in range(Nx):
    cd[i,i] = a
dr = np.zeros((Nx,Ny))
dl = np.zeros((Nx,Ny))
du = np.zeros((Nx,Ny))
dd = np.zeros((Nx,Ny))
for i in range(Nx):
    dd[i-1,i] = b
x = mps_opt.MPS_OPT(N=[Nx,Ny],
                    hamType = 'sep_2d',
                    periodic_x = True,
                    periodic_y = True,
                    plotExpVals = True,
                    plotConv = True,
                    hamParams = (jl,jr,jd,ju,cr,cl,cd,cu,dr,dl,dd,du,[0,-s]))
E = x.kernel()
