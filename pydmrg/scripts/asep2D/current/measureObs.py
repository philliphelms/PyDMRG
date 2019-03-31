import numpy as np
from sys import argv
from tools.aux.process_states import *
from mpo.asep2D import *

# Get directory name
folder = argv[1]#'saved_states/multiLane_Nx2Ny10mbd10_1548458158/'
bcs = argv[2]
# Load results file to get s
npzfile = np.load(folder+'results.npz')
s = npzfile['s']
Eoriginal = npzfile['E']
N = int(npzfile['Nx'])*int(npzfile['Ny'])
Nx = int(npzfile['Nx'])
Ny = int(npzfile['Ny'])
nStates = 2
p = 0.1

# If periodic, then say so
if bcs == 'periodic':
    periodicy = False
    periodicx = True
else:
    periodicy = False
    periodicx = False

# Allocate data structures
E           = np.zeros((len(s),2))
EE          = np.zeros((len(s),2,N-1))
EEorth      = np.zeros((len(s),2,N-1))
EEl         = np.zeros((len(s),2,N-1))
EElorth     = np.zeros((len(s),2,N-1))
rho         = np.zeros((len(s),Nx,Ny))
rhoOrth     = np.zeros((len(s),Nx,Ny))
actVert     = np.zeros((len(s),Nx,Ny+1))
actVertOrth = np.zeros((len(s),Nx,Ny+1))
actHorz     = np.zeros((len(s),Nx+1,Ny))
actHorzOrth = np.zeros((len(s),Nx+1,Ny))
curVert     = np.zeros((len(s),Nx,Ny+1))
curVertOrth = np.zeros((len(s),Nx,Ny+1))
curHorz     = np.zeros((len(s),Nx+1,Ny))
curHorzOrth = np.zeros((len(s),Nx+1,Ny))

#for i in range(len(s)-1,0,-1):
for i in range(len(s)):#-1,-1,-1):
    print('{}/{}'.format(i+1,len(s)))
    print(s[i])
    
    # Specify new file name
    mps_fname = folder + 'MPS_s'+str(i)+'_mbd0'
    lmps_fname= folder + 'MPS_s'+str(i)+'_mbd0_left'
    
    # Get correct model parameters
    if bcs == 'closed':
        #hamParams = np.array([0.5,0.5,p,1.-p,0.0,0.0,0.5,0.5,0.0,0.0,0.5,0.5,0.,s[i]])
        hamParams = np.array([p,1.-p,0.5,0.5,0.0,0.0,0.5,0.5,0.0,0.0,0.5,0.5,s[i],0.])
        mpo = return_mpo((Nx,Ny),hamParams,periodicy=periodicy,periodicx=periodicx)
    elif bcs == 'open':
        hamParams = np.array([0.5,0.5,p,1.-p,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.,s[i]])
        #hamParams = np.array([p ,1.-p,0.5,0.5,0.5,0.5,0.9,0.1,0.5,0.5,0.9,0.1,s[i],0.])
        mpo = return_mpo((Nx,Ny),hamParams,periodicy=periodicy,periodicx=periodicx)
    elif bcs == 'periodic':
        hamParams = np.array([0.5,0.5,p,1.-p,0.0,0.0,0.5,0.5,0.0,0.0,0.5,0.5,0.,s[i]])
        mpo = return_mpo((Nx,Ny),hamParams,periodicy=periodicy,periodicx=periodicx)
    
    # Calculate Energy
    for state in range(nStates):
        E[i,state] = contract(mps=mps_fname,mpo=mpo,state=state)/contract(mps=mps_fname,state=state)
    print('E',E[i,:])
    print('E (DMRG) = {}'.format(Eoriginal[i]))

    # Calculate Entanglement Entropies
    EE     [i,:,:] = calc_entanglement_all(mps_fname ,orth=False)
    print(EE[i,:,:])
    print('Printing EE')
    print(EE.shape)
    nx,ny,nz = EE.shape
    for q in range(nz):
        print(EE[i,0,q])
    print('EE',EE[i,0,int(N/2)])
    EEl    [i,:,:] = calc_entanglement_all(lmps_fname,orth=False)
    print('EEl',EEl[i,0,int(N/2)])
    EEorth [i,:,:] = calc_entanglement_all(mps_fname ,orth=True)
    print('EEorth',EEorth[i,0,int(N/2)])
    EElorth[i,:,:] = calc_entanglement_all(lmps_fname,orth=True)
    print('EElorth',EElorth[i,0,int(N/2)])

    # Calculate Densities
    rhoVec = calc_density_all(mps_fname,lmps_fname,orth=False,state=0)
    rho    [i,:,:] = np.reshape(rhoVec,(Nx,Ny))
    print('rho',rho[i,:,:])
    rhoOrthVec = 0.5*(calc_density_all(mps_fname,lmps_fname,orth=True,state=0) + calc_density_all(mps_fname,lmps_fname,orth=True,state=1))
    rhoOrth[i,:,:] = np.reshape(rhoOrthVec,(Nx,Ny))
    print('rhoOrth',rhoOrth[i,:,:])
    
    """
    # Calculate Total Vertical Activity
    mpo = act_mpo((Nx,Ny),hamParams,includex=False)
    print('Vertical Total Activity = {}'.format(contract(mps=mps_fname,lmps=lmps_fname,mpo=mpo,orth=False)/contract(mps=mps_fname,lmps=lmps_fname,orth=False)))
    # Calculate Total Horizontal Activity
    mpo = act_mpo((Nx,Ny),hamParams,includey=False)
    print('Horizontal Total Activity = {}'.format(contract(mps=mps_fname,lmps=lmps_fname,mpo=mpo,orth=False)/contract(mps=mps_fname,lmps=lmps_fname,orth=False)))
    # Calculate Total Vertical Current
    mpo = curr_mpo((Nx,Ny),hamParams,includex=False)
    print('Vertical Total Current = {}'.format(contract(mps=mps_fname,lmps=lmps_fname,mpo=mpo,orth=False)/contract(mps=mps_fname,lmps=lmps_fname,orth=False)))
    # Calculate Total Horizontal Current
    mpo = curr_mpo((Nx,Ny),hamParams,includey=False)
    print('Horizontal Total Current = {}'.format(contract(mps=mps_fname,lmps=lmps_fname,mpo=mpo,orth=False)/contract(mps=mps_fname,lmps=lmps_fname,orth=False)))

    # Calculate Horizontal Activities
    for xind in range(Nx+1):
        for yind in range(Ny):
            if xind == 0:
                mpo = act_mpo((Nx,Ny),hamParams,singleBond=True,ybond=yind,xbond='left',orientation='horz')
                print('Left Side')
            elif xind == Nx:
                mpo = act_mpo((Nx,Ny),hamParams,singleBond=True,ybond=yind,xbond='right',orientation='horz')
                print('Right Side')
            else:
                mpo = act_mpo((Nx,Ny),hamParams,singleBond=True,ybond=yind,xbond=xind-1,orientation='horz')
                print('Site {}'.format(xind-1))
            actHorz    [i,xind,yind] = contract(mps=mps_fname,lmps=lmps_fname,mpo=mpo,orth=False)/contract(mps=mps_fname,lmps=lmps_fname,orth=False)
            actHorzOrth[i,xind,yind] = 0.5*(contract(mps=mps_fname,lmps=lmps_fname,mpo=mpo,orth=True,state=0)/contract(mps=mps_fname,lmps=lmps_fname,orth=True,state=0) +\
                                            contract(mps=mps_fname,lmps=lmps_fname,mpo=mpo,orth=True,state=1)/contract(mps=mps_fname,lmps=lmps_fname,orth=True,state=1))
            print(actHorz[i,xind,yind])
    print('actHorz',actHorz[i,:,:])
    print('actHorzOrth',actHorzOrth[i,:,:])
    print('Single Site Activity Horz = {}'.format(np.sum(np.sum(actHorz[i,:,:]))))

    # Calculate Vertical Activities
    for xind in range(Nx):
        for yind in range(Ny+1):
            if yind == 0:
                mpo = act_mpo((Nx,Ny),hamParams,singleBond=True,xbond=xind,ybond='bottom',orientation='vert')
            elif yind == Ny:
                mpo = act_mpo((Nx,Ny),hamParams,singleBond=True,xbond=xind,ybond='top',orientation='vert')
            else:
                mpo = act_mpo((Nx,Ny),hamParams,singleBond=True,xbond=xind,ybond=yind-1,orientation='vert')
            actVert    [i,xind,yind] = contract(mps=mps_fname,lmps=lmps_fname,mpo=mpo,orth=False)/contract(mps=mps_fname,lmps=lmps_fname,orth=False)
            actVertOrth[i,xind,yind] = 0.5*(contract(mps=mps_fname,lmps=lmps_fname,mpo=mpo,orth=True,state=0)/contract(mps=mps_fname,lmps=lmps_fname,orth=True,state=0) + \
                                            contract(mps=mps_fname,lmps=lmps_fname,mpo=mpo,orth=True,state=1)/contract(mps=mps_fname,lmps=lmps_fname,orth=True,state=1))
    print('actVert',actVert[i,:,:])
    print('actVertOrth',actVertOrth[i,:,:])
    print('Single Site Activity Vert = {}'.format(np.sum(np.sum(actVert[i,:,:]))))

    # Calculate Vertical Currents
    for xind in range(Nx):
        for yind in range(Ny+1):
            if yind == 0:
                mpo = curr_mpo((Nx,Ny),hamParams,singleBond=True,xbond=xind,ybond='bottom',orientation='vert')
            elif yind == Ny:
                mpo = curr_mpo((Nx,Ny),hamParams,singleBond=True,xbond=xind,ybond='top',orientation='vert')
            else:
                mpo = curr_mpo((Nx,Ny),hamParams,singleBond=True,xbond=xind,ybond=yind-1,orientation='vert')
            curVert    [i,xind,yind] = contract(mps=mps_fname,lmps=lmps_fname,mpo=mpo,orth=False)/contract(mps=mps_fname,lmps=lmps_fname,orth=False)
            curVertOrth[i,xind,yind] = 0.5*(contract(mps=mps_fname,lmps=lmps_fname,mpo=mpo,orth=True,state=0)/contract(mps=mps_fname,lmps=lmps_fname,orth=True,state=0) +\
                                            contract(mps=mps_fname,lmps=lmps_fname,mpo=mpo,orth=True,state=1)/contract(mps=mps_fname,lmps=lmps_fname,orth=True,state=1))
    print('curVert',curVert[i,:,:])
    print('curVertOrth',curVertOrth[i,:,:])
    print('Single Site Current Vert = {}'.format(np.sum(np.sum(curVert[i,:,:]))))

    # Calculate Horizontal Currents
    for xind in range(Nx+1):
        for yind in range(Ny):
            if xind == 0:
                mpo = curr_mpo((Nx,Ny),hamParams,singleBond=True,ybond=yind,xbond='left',orientation='horz')
            elif xind == Nx:
                mpo = curr_mpo((Nx,Ny),hamParams,singleBond=True,ybond=yind,xbond='right',orientation='horz')
            else:
                mpo = curr_mpo((Nx,Ny),hamParams,singleBond=True,xbond=xind-1,ybond=yind,orientation='horz')
            curHorz    [i,xind,yind] = contract(mps=mps_fname,lmps=lmps_fname,mpo=mpo,orth=False)/contract(mps=mps_fname,lmps=lmps_fname,orth=False)
            curHorzOrth[i,xind,yind] = 0.5*(contract(mps=mps_fname,lmps=lmps_fname,mpo=mpo,orth=True,state=0)/contract(mps=mps_fname,lmps=lmps_fname,orth=True,state=0) +\
                                            contract(mps=mps_fname,lmps=lmps_fname,mpo=mpo,orth=True,state=1)/contract(mps=mps_fname,lmps=lmps_fname,orth=True,state=1))
    print('curHorz',curHorz[i,:,:])
    print('curHorzOrth',curHorz[i,:,:])
    print('Single Site Current Horz = {}'.format(np.sum(np.sum(curHorz[i,:,:]))))
    """
    # Save Results
    np.savez(folder+'observables.npz',s=s,
                                      E=E,
                                      EE=EE,
                                      EEorth=EEorth,
                                      EEl=EEl,
                                      EElorth=EElorth,
                                      rho=rho,
                                      rhoOrth=rhoOrth)
                                      #actVert=actVert,
                                      #actVertOrth=actVertOrth,
                                      #actHorz=actHorz,
                                      #actHorzOrth=actHorzOrth,
                                      #curVert=curVert,
                                      #curVertOrth=curVertOrth,
                                      #curHorz=curHorz,
                                      #curHorzOrth=curHorzOrth)
