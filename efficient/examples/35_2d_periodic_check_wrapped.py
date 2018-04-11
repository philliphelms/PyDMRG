import mps_opt
import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Run a check to ensure PBC are working, by setting up staggered
# SEPs on a 2D lattice in both directions.
#-----------------------------------------------------------------------------

# Run 1D for comparison
N = 6
a = 2/3
b = 0.35
s = 0
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
    dr[i,i-2] = b
dl = np.zeros((Nx,Ny))
du = np.zeros((Nx,Ny))
dd = np.zeros((Nx,Ny))
print(jr)
print(cr)
print(dr)
x = mps_opt.MPS_OPT(N=[Nx,Ny],
                    hamType = 'sep_2d',
                    periodic_x = True,
                    periodic_y = True,
                    plotExpVals = True,
                    plotConv = True,
                    hamParams = (jl,jr,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,[s,0]))
E = x.kernel()
"""
# Run 2D in x-direction
Nx = N
Ny = N
jl = np.zeros((Nx,Ny))
jr = np.ones((Nx,Ny))
for i in range(Ny):
    jr[i,:i] = 0
jr[:,-1] = 0
ju = np.zeros((Nx,Ny))
jd = np.zeros((Nx,Ny))
cr = np.zeros((Nx,Ny))
for i in range(Ny):
    cr[i,i] = a
cl = np.zeros((Nx,Ny))
cu = np.zeros((Nx,Ny))
cd = np.zeros((Nx,Ny))
dr = np.zeros((Nx,Ny))
dr[:,-1] = b
dl = np.zeros((Nx,Ny))
du = np.zeros((Nx,Ny))
dd = np.zeros((Nx,Ny))
print(jr)
print(cr)
print(dr)
x = mps_opt.MPS_OPT(N=[Nx,Ny],
                    hamType = 'sep_2d',
                    periodic_x = True,
                    periodic_y = True,
                    plotExpVals = True,
                    plotConv = True,
                    hamParams = (jl,jr,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,[s,0]))
E = x.kernel()
"""

"""
# Run 2D in backwards x-direction
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
                    hamParams = (jl,jr,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,[-s,0]))
E = x.kernel()

# Run 2D in y-direction
Nx = N
Ny = N
jl = np.zeros((Nx,Ny))
jr = np.zeros((Nx,Ny))
jd = np.ones((Nx,Ny))
jd[-1,:] = 0
ju = np.zeros((Nx,Ny))
cr = np.zeros((Nx,Ny))
cl = np.zeros((Nx,Ny))
cd = np.zeros((Nx,Ny))
cu = np.zeros((Nx,Ny))
cd[0,:] = a
dr = np.zeros((Nx,Ny))
dl = np.zeros((Nx,Ny))
dd = np.zeros((Nx,Ny))
du = np.zeros((Nx,Ny))
dd[-1,:] = b
x = mps_opt.MPS_OPT(N=[Nx,Ny],
                    hamType = 'sep_2d',
                    periodic_x = True,
                    periodic_y = True,
                    plotExpVals = True,
                    plotConv = True,
                    add_noise = False,
                    hamParams = (jl,jr,jd,ju,cr,cl,cd,cu,dr,dl,dd,du,[0,-s]))
E = x.kernel()
# Run 2D in y-direction
Nx = N
Ny = N
jl = np.zeros((Nx,Ny))
jr = np.zeros((Nx,Ny))
ju = np.ones((Nx,Ny))
ju[0,:] = 0
jd = np.zeros((Nx,Ny))
cr = np.zeros((Nx,Ny))
cl = np.zeros((Nx,Ny))
cd = np.zeros((Nx,Ny))
cu = np.zeros((Nx,Ny))
cu[-1,:] = a
dr = np.zeros((Nx,Ny))
dl = np.zeros((Nx,Ny))
dd = np.zeros((Nx,Ny))
du = np.zeros((Nx,Ny))
du[0,:] = b
x = mps_opt.MPS_OPT(N=[Nx,Ny],
                    hamType = 'sep_2d',
                    periodic_x = True,
                    periodic_y = True,
                    plotExpVals = True,
                    plotConv = True,
                    add_noise = False,
                    hamParams = (jl,jr,jd,ju,cr,cl,cd,cu,dr,dl,dd,du,[0,s]))
E = x.kernel()
"""
