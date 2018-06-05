import mps_opt
import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Run a check to ensure PBC are working, by setting up staggered
# SEPs on a 2D lattice in both directions.
#-----------------------------------------------------------------------------

###################################################
# Run 1D for comparison
N = 5
Nx = 5
Ny = 2
a = 2/3
b = 0.35
s = -1
dividing_points = np.array([2,4])#,8,10,12,14])
x = mps_opt.MPS_OPT(N=N,
                    hamType = "sep",
                    plotExpVals = True,
                    hamParams = (a,0,1,0,0,b,s))
E = x.kernel()
print('Expected Energy Result = {}'.format(E*N))
####################################################
# Run 2D in x-direction
if True:
    jl = np.zeros((Ny,Nx))
    jr = np.ones((Ny,Nx))
    for i in range(len(dividing_points)):
        jr[i,dividing_points[i]-1] = 0
    ju = np.zeros((Ny,Nx))
    jd = np.zeros((Ny,Nx))
    cr = np.zeros((Ny,Nx))
    for i in range(len(dividing_points)):
        cr[i,dividing_points[i]] = a
    cl = np.zeros((Ny,Nx))
    cu = np.zeros((Ny,Nx))
    cd = np.zeros((Ny,Nx))
    dr = np.zeros((Ny,Nx))
    for i in range(len(dividing_points)):
        dr[i,dividing_points[i]-1] = b
    dl = np.zeros((Ny,Nx))
    du = np.zeros((Ny,Nx))
    dd = np.zeros((Ny,Nx))
    x = mps_opt.MPS_OPT(N=[Nx,Ny],
                        hamType = 'sep_2d',
                        periodic_x = True,
                        periodic_y = True,
                        plotExpVals = True,
                        plotConv = True,
                        hamParams = (jl,jr,jd,ju,cr,cl,cd,cu,dr,dl,dd,du,[s,0]))
    E = x.kernel()
########################################################
# Run 2D in reverse x-direction
if False:
    jl = np.ones((Ny,Nx))
    for i in range(len(dividing_points)):
        jl[i,dividing_points[i]] = 0
    jr = np.zeros((Ny,Nx))
    ju = np.zeros((Ny,Nx))
    jd = np.zeros((Ny,Nx))
    cr = np.zeros((Ny,Nx))
    cl = np.zeros((Ny,Nx))
    for i in range(len(dividing_points)):
        cl[i,dividing_points[i]-1] = a
    cu = np.zeros((Ny,Nx))
    cd = np.zeros((Ny,Nx))
    dr = np.zeros((Ny,Nx))
    dl = np.zeros((Ny,Nx))
    for i in range(len(dividing_points)):
        dl[i,dividing_points[i]] = b
    du = np.zeros((Ny,Nx))
    dd = np.zeros((Ny,Nx))
    x = mps_opt.MPS_OPT(N=[Nx,Ny],
                        hamType = 'sep_2d',
                        periodic_x = True,
                        periodic_y = True,
                        plotExpVals = True,
                        plotConv = True,
                        hamParams = (jl,jr,jd,ju,cr,cl,cd,cu,dr,dl,dd,du,[-s,0]))
    E = x.kernel()
########################################################
# Run 2D in y-direction
if False:
    jl = np.zeros((Nx,Ny))
    jr = np.zeros((Nx,Ny))
    ju = np.ones((Nx,Ny))
    for i in range(len(dividing_points)):
        ju[dividing_points[i],i] = 0
    jd = np.zeros((Nx,Ny))
    cr = np.zeros((Nx,Ny))
    cl = np.zeros((Nx,Ny))
    cu = np.zeros((Nx,Ny))
    for i in range(len(dividing_points)):
        cu[dividing_points[i]-1,i] = a
    cd = np.zeros((Nx,Ny))
    dr = np.zeros((Nx,Ny))
    dl = np.zeros((Nx,Ny))
    du = np.zeros((Nx,Ny))
    for i in range(len(dividing_points)):
        du[dividing_points[i],i] = b
    dd = np.zeros((Ny,Nx))
    x = mps_opt.MPS_OPT(N=[Ny,Nx],
                        hamType = 'sep_2d',
                        periodic_x = True,
                        periodic_y = True,
                        plotExpVals = True,
                        plotConv = True,
                        hamParams = (jl,jr,jd,ju,cr,cl,cd,cu,dr,dl,dd,du,[0,s]))
    E = x.kernel()
############################################################
# Run 2D in reverse y-direction
if True:
    jl = np.zeros((Nx,Ny))
    jr = np.zeros((Nx,Ny))
    ju = np.zeros((Nx,Ny))
    jd = np.ones((Nx,Ny))
    for i in range(len(dividing_points)):
        jd[dividing_points[i]-1,i] = 0
    cr = np.zeros((Nx,Ny))
    cl = np.zeros((Nx,Ny))
    cu = np.zeros((Nx,Ny))
    cd = np.zeros((Nx,Ny))
    for i in range(len(dividing_points)):
        cd[dividing_points[i],i] = a
    dr = np.zeros((Nx,Ny))
    dl = np.zeros((Nx,Ny))
    du = np.zeros((Nx,Ny))
    dd = np.zeros((Nx,Ny))
    for i in range(len(dividing_points)):
        dd[dividing_points[i]-1,i] = b
    x = mps_opt.MPS_OPT(N=[Ny,Nx],
                        hamType = 'sep_2d',
                        periodic_x = True,
                        periodic_y = True,
                        plotExpVals = True,
                        plotConv = True,
                        hamParams = (jl,jr,jd,ju,cr,cl,cd,cu,dr,dl,dd,du,[0,-s]))
    E = x.kernel()
