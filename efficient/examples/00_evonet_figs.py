import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm # Colormaps

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='serif')
plt.rcParams['text.latex.unicode']=False
plt.style.use('seaborn-bright')#'classic')#'fivethirtyeight') #'fivethirtyeight'
mpl.rc('grid',color = 'k')
mpl.rc('grid',linestyle = ':')
mpl.rc('grid',linewidth = 0.5)
mpl.rc('lines', linewidth=3.)
mpl.rc('xtick', labelsize=16)
mpl.rc('ytick', labelsize=16)
mpl.rc('axes', titlesize=20)
mpl.rc('axes', labelsize=16)
mpl.rc('font',**{'family':'serif','serif':['Arial']})
mpl.rc('legend', fontsize = 16)
#mpl.rcParams['figure.figsize'] = [5.0, 4.0]
colormap = cm.plasma #coolwarm, inferno, viridis


###############################################
# Make Phase Diagram & Example Plots

# 1 - Phase Diagram
if False:
    plt.figure()
    p = np.linspace(0,1,10)
    s = np.linspace(-5,5,10)
    for i in range(len(p)):
        for j in range(len(s)):
            a = 1/(2*0.5)*((p[i]-(1-p[i])-0.5+0.5)+np.sqrt((p[i]-(1-p[i])-0.5+0.5)**2+4*0.5*0.5))
            b = 1/(2*0.5)*((p[i]-(1-p[i])-0.5+0.5)+np.sqrt((p[i]-(1-p[i])-0.5+0.5)**2+4*0.5*0.5))
            rho_a = 1/(1+a) # Note that rho_a = 1-rho_b for all values of p
            rho_b = b/(1+b)
            u = 1/(1+np.exp(s[j]))
            # Maximal Current
            if (rho_a > 0.5) and (rho_b < 0.5):
                if (u < 0.5):
                    plt.plot(p[i],s[j],'go')
            if (rho_a > 0.5) and (rho_b > 0.5):
                if u < rho_a:
                    plt.plot(p[i],s[j],'go')
            if (rho_a < 0.5) and (rho_b < 0.5):
                if u < 1-rho_b:
                    plt.plot(p[i],s[j],'go')
            if (rho_a < 0.5) and (rho_b > 0.5):
                if u < rho_a*(1-rho_b)/(rho_b+rho_a-2*rho_a*rho_b):
                    plt.plot(p[i],s[j],'go')
            # HD/LD
            if rho_a > 0.5:
                if u > rho_a**2/(1-2*rho_a+2*rho_a**2):
                    plt.plot(p[i],s[j],'co')
            if rho_a < 0.5:
                if u > 0.5:
                    plt.plot(p[i],s[j],'co')
            # Anti Shock
            if u > 0.5:
                if (rho_a > 0.5) and (rho_a > 1-rho_b) and (u < rho_a**2/(1-2*rho_a+2*rho_a**2)):
                    plt.plot(p[i],s[j],'ro')
                elif (rho_b < 0.5) and (rho_a < 1-rho_b) and (u < (1-rho_b)**2/(1-2*rho_b+2*rho_b**2)):
                    plt.plot(p[i],s[j],'ro')
            # Shock
            if (rho_a < 0.5) and (rho_b > 0.5) and (u < rho_a*rho_b/(1-rho_a-rho_b+2*rho_a*rho_b)):
                if (u < (1-rho_a)*(1-rho_b)/(1-rho_b-rho_a+2*rho_a*rho_b)):
                    if (u > rho_a*(1-rho_b)/(rho_b+rho_a-2*rho_a*rho_b)):
                        plt.plot(p[i],s[j],'bo')

if False:
    npts = 10000
    plt.figure()
    # Center Line
    plt.plot(np.array([0,1]),np.array([0,0]),'k-',linewidth=3)
    # Shock Dividing Line
    p = np.linspace(0,1,npts)
    a = 1/(2*0.5)*((p-(1-p)-0.5+0.5)+np.sqrt((p-(1-p)-0.5+0.5)**2+4*0.5*0.5))
    b = 1/(2*0.5)*((p-(1-p)-0.5+0.5)+np.sqrt((p-(1-p)-0.5+0.5)**2+4*0.5*0.5))
    rho_a = 1/(1+a)
    rho_b = b/(1+b)
    u = rho_a**2/(1-2*rho_a+2*rho_a**2)
    print(u)
    #mu = 
    plt.plot(p,np.log(1/y-1),'k-',linewidth=3)
    p = np.linspace(0,0.5,npts)
    #plt.fill_between(np.zeros(p.shape),

###############################################
# Phase Sketches
if False:
    fig = plt.figure()
    ax = fig.gca()
    npts = 1000
    x1 = np.linspace(0.1,0.8,npts)
    y1 = 0.65*np.ones(x1.shape)
    x2 = np.linspace(0.2,0.9,npts)
    y2 = 0.35*np.ones(x2.shape)
    x3 = np.linspace(0,0.1,npts)
    y3 = np.sqrt(0.1**2-(x3-0.1)**2)+0.55
    x4 = np.linspace(0.8,1,npts)
    y4 = np.sqrt(0.2**2-(x4-0.8)**2)+0.45
    x5 = np.linspace(0,0.2,npts)
    y5 = -np.sqrt(0.2**2-(x5-0.2)**2)+0.55
    x6 = np.linspace(0.9,1,npts)
    y6 = -np.sqrt(0.1**2-(x6-0.9)**2)+0.45
    plt.plot(x1,y1,'r',label='High Density')
    plt.plot(x2,y2,'b',label='Low Density')
    plt.plot(x3,y3,'r')
    plt.plot(x4,y4,'r')
    plt.plot(x5,y5,'b')
    plt.plot(x6,y6,'b')
    # Create Average
    x1 = np.linspace(0,0.1,npts)
    y1 = ((np.sqrt(0.1**2-(x1-0.1)**2)+0.55)+(-np.sqrt(0.2**2-(x1-0.2)**2)+0.55))/2.
    x2 = np.linspace(0.1,0.2,npts)
    y2 = (0.65-np.sqrt(0.2**2-(x2-0.2)**2)+0.55)/2.
    x3 = np.linspace(0.2,0.8,npts)
    y3 = (0.65+0.35)/2.*np.ones(x3.shape)
    x4 = np.linspace(0.8,0.9,npts)
    y4 = (0.45+np.sqrt(0.2**2-(x4-0.8)**2)+0.35)/2.
    x5 = np.linspace(0.9,1.,npts)
    y5 = (-np.sqrt(0.1**2-(x5-0.9)**2)+0.45+np.sqrt(0.2**2-(x5-0.8)**2)+0.45)/2.
    plt.plot(x1,y1,'b',label='Average')
    plt.plot(x2,y2,'b')
    plt.plot(x3,y3,'b')
    plt.plot(x4,y4,'b')
    plt.plot(x5,y5,'b')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.legend()
    
if False:
    fig = plt.figure()
    ax = fig.gca()
    npts = 1000
    x1 = np.linspace(0.1,0.9,npts)
    y1 = 0.065*np.arctan(100*(x1-0.5))+0.5
    plt.plot(x1,y1,'r',label='Possible Profiles')
    x1 = np.linspace(0.1,0.9,npts)
    y1 = 0.065*np.arctan(100*(x1-0.3))+0.5
    plt.plot(x1,y1,'r')
    x1 = np.linspace(0.1,0.9,npts)
    y1 = 0.065*np.arctan(100*(x1-0.4))+0.5
    plt.plot(x1,y1,'r')
    x1 = np.linspace(0.1,0.9,npts)
    y1 = 0.065*np.arctan(100*(x1-0.6))+0.5
    plt.plot(x1,y1,'r')
    x1 = np.linspace(0.1,0.9,npts)
    y1 = 0.065*np.arctan(100*(x1-0.7))+0.5
    plt.plot(x1,y1,'r')
    x1 = np.linspace(0,0.1,npts)
    y1 = np.sqrt(0.1**2-(x1-0.1)**2)+0.3
    x2 = np.linspace(0.1,0.9,npts)
    y2 = (0.6-0.4)/(0.9-0.1)*(x2-0.1)+0.4
    x3 = np.linspace(0.9,1.,npts)
    y3 = -np.sqrt(0.1**2-(x3-0.9)**2)+0.7
    plt.plot(x1,y1,'b',label='Average')
    plt.plot(x2,y2,'b')
    plt.plot(x3,y3,'b')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.legend()

if False:
    fig = plt.figure()
    ax = fig.gca()
    npts = 1000
    x1 = np.linspace(0.2,0.8,npts)
    y1 = 0.5*np.ones(x1.shape)
    plt.plot(x1,y1,'b')
    x1 = np.linspace(0.,0.2,npts)
    y1 = -np.sqrt(0.2**2-(x1-0.2)**2)+0.7
    plt.plot(x1,y1,'b')
    x1 = np.linspace(0.8,1.,npts)
    y1 = np.sqrt(0.2**2-(x1-0.8)**2)+0.3
    plt.plot(x1,y1,'b')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.legend()

if True:
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(5.5, 6.5))
    c_map_1 = 0.2
    c_map_2 = 0.8
    print(colormap(c_map_1))
    print(colormap(c_map_2))
    npts = 1000
    x1 = np.linspace(0.2,0.8,npts)
    y1 = 0.5*np.ones(x1.shape)
    ax1.plot(x1,y1,color=colormap(c_map_1))
    x1 = np.linspace(0.,0.2,npts)
    y1 = -np.sqrt(0.2**2-(x1-0.2)**2)+0.7
    ax1.plot(x1,y1,color=colormap(c_map_1))
    x1 = np.linspace(0.8,1.,npts)
    y1 = np.sqrt(0.2**2-(x1-0.8)**2)+0.3
    ax1.plot(x1,y1,color=colormap(c_map_1))
    ax1.set_xlim(0,1)
    ax1.set_ylim(0.3,0.7)
    npts = 1000
    x1 = np.linspace(0.1,0.8,npts)
    y1 = 0.65*np.ones(x1.shape)
    x2 = np.linspace(0.2,0.9,npts)
    y2 = 0.35*np.ones(x2.shape)
    x3 = np.linspace(0,0.1,npts)
    y3 = np.sqrt(0.1**2-(x3-0.1)**2)+0.55
    x4 = np.linspace(0.8,1,npts)
    y4 = np.sqrt(0.2**2-(x4-0.8)**2)+0.45
    x5 = np.linspace(0,0.2,npts)
    y5 = -np.sqrt(0.2**2-(x5-0.2)**2)+0.55
    x6 = np.linspace(0.9,1,npts)
    y6 = -np.sqrt(0.1**2-(x6-0.9)**2)+0.45
    ax2.plot(x1,y1,':',color=colormap(c_map_2),linewidth=2,label='Possible Density Profiles')
    ax2.plot(x2,y2,':',color=colormap(c_map_2),linewidth=2)
    ax2.plot(x3,y3,':',color=colormap(c_map_2),linewidth=2)
    ax2.plot(x4,y4,':',color=colormap(c_map_2),linewidth=2)
    ax2.plot(x5,y5,':',color=colormap(c_map_2),linewidth=2)
    ax2.plot(x6,y6,':',color=colormap(c_map_2),linewidth=2)
    # Create Average
    x1 = np.linspace(0,0.1,npts)
    y1 = ((np.sqrt(0.1**2-(x1-0.1)**2)+0.55)+(-np.sqrt(0.2**2-(x1-0.2)**2)+0.55))/2.
    x2 = np.linspace(0.1,0.2,npts)
    y2 = (0.65-np.sqrt(0.2**2-(x2-0.2)**2)+0.55)/2.
    x3 = np.linspace(0.2,0.8,npts)
    y3 = (0.65+0.35)/2.*np.ones(x3.shape)
    x4 = np.linspace(0.8,0.9,npts)
    y4 = (0.45+np.sqrt(0.2**2-(x4-0.8)**2)+0.35)/2.
    x5 = np.linspace(0.9,1.,npts)
    y5 = (-np.sqrt(0.1**2-(x5-0.9)**2)+0.45+np.sqrt(0.2**2-(x5-0.8)**2)+0.45)/2.
    ax2.plot(x1,y1,color=colormap(c_map_1),label='Average')
    ax2.plot(x2,y2,color=colormap(c_map_1))
    ax2.plot(x3,y3,color=colormap(c_map_1))
    ax2.plot(x4,y4,color=colormap(c_map_1))
    ax2.plot(x5,y5,color=colormap(c_map_1))
    ax2.set_xlim(0,1)
    ax2.set_ylim(0.3,0.7)
    #ax2.legend(loc=1)
    npts = 1000
    x1 = np.linspace(0.1,0.9,npts)
    y1 = 0.065*np.arctan(100*(x1-0.5))+0.5
    ax3.plot(x1,y1,':',color=colormap(c_map_2),linewidth=2,label='Possible Profiles')
    x1 = np.linspace(0.1,0.9,npts)
    y1 = 0.065*np.arctan(100*(x1-0.3))+0.5
    ax3.plot(x1,y1,':',color=colormap(c_map_2),linewidth=2)
    x1 = np.linspace(0.1,0.9,npts)
    y1 = 0.065*np.arctan(100*(x1-0.4))+0.5
    ax3.plot(x1,y1,':',linewidth=2,color=colormap(c_map_2))
    x1 = np.linspace(0.1,0.9,npts)
    y1 = 0.065*np.arctan(100*(x1-0.6))+0.5
    ax3.plot(x1,y1,':',linewidth=2,color=colormap(c_map_2))
    x1 = np.linspace(0.1,0.9,npts)
    y1 = 0.065*np.arctan(100*(x1-0.7))+0.5
    ax3.plot(x1,y1,':',linewidth=2,color=colormap(c_map_2))
    x1 = np.linspace(0,0.1,npts)
    y1 = np.sqrt(0.1**2-(x1-0.1)**2)+0.3
    x2 = np.linspace(0.1,0.9,npts)
    y2 = (0.6-0.4)/(0.9-0.1)*(x2-0.1)+0.4
    x3 = np.linspace(0.9,1.,npts)
    y3 = -np.sqrt(0.1**2-(x3-0.9)**2)+0.7
    ax3.plot(x1,y1,color=colormap(c_map_1),label='Average')
    ax3.plot(x2,y2,color=colormap(c_map_1))
    ax3.plot(x3,y3,color=colormap(c_map_1))
    ax3.set_xlim(0,1)
    ax3.set_ylim(0.3,0.7)

    npts = 1000
    x1 = np.linspace(0.1,0.9,npts)
    y1 = -0.065*np.arctan(100*(x1-0.5))+0.5
    ax4.plot(x1,y1,':',linewidth=2,color=colormap(c_map_2),label='Possible Profiles')
    x1 = np.linspace(0.1,0.9,npts)
    y1 = -0.065*np.arctan(100*(x1-0.3))+0.5
    ax4.plot(x1,y1,':',linewidth=2,color=colormap(c_map_2))
    x1 = np.linspace(0.1,0.9,npts)
    y1 = -0.065*np.arctan(100*(x1-0.4))+0.5
    ax4.plot(x1,y1,':',linewidth=2,color=colormap(c_map_2))
    x1 = np.linspace(0.1,0.9,npts)
    y1 = -0.065*np.arctan(100*(x1-0.6))+0.5
    ax4.plot(x1,y1,':',linewidth=2,color=colormap(c_map_2))
    x1 = np.linspace(0.1,0.9,npts)
    y1 = -0.065*np.arctan(100*(x1-0.7))+0.5
    ax4.plot(x1,y1,':',linewidth=2,color=colormap(c_map_2))
    x1 = np.linspace(0,0.1,npts)
    y1 = -np.sqrt(0.1**2-(x1-0.1)**2)+0.7
    x2 = np.linspace(0.1,0.9,npts)
    y2 = -(0.6-0.4)/(0.9-0.1)*(x2-0.1)+0.6
    x3 = np.linspace(0.9,1.,npts)
    y3 = np.sqrt(0.1**2-(x3-0.9)**2)+0.3
    ax4.plot(x1,y1,color=colormap(c_map_1),label='Average')
    ax4.plot(x2,y2,color=colormap(c_map_1))
    ax4.plot(x3,y3,color=colormap(c_map_1))
    ax4.set_xlim(0,1)
    ax4.set_ylim(0.3,0.7)
    fig.subplots_adjust(hspace=0.5)
    #ax1.text(0.1,0.6,'Maximal Current')
    #ax2.text(0.1,0.6,'HD/LD Coexistence')#,backgroundcolor='w')
    #ax3.text(0.1,0.6,'Shock')#,backgroundcolor='w')
    #ax4.text(0.1,0.6,'Anti-Shock')#,backgroundcolor='w')
    ax1.set_title('Maximal Current',fontweight='bold')
    ax2.set_title('HD/LD Coexistence')#,backgroundcolor='w')
    ax3.set_title('Shock')#,backgroundcolor='w')
    ax4.set_title('Anti-Shock')#,backgroundcolor='w')

    plt.savefig('phases.pdf')

    

###############################################
# 1D Phase Diagram
filename = 'asep_original_N10_data_p500s500.npz'

npzfile = np.load(filename)
print(npzfile.files)
CGF = npzfile['CGF']
EE = npzfile['EE']
nPart = npzfile['nPart']
density = npzfile['density']
s = np.linspace(-5,5,len(CGF))
p = np.linspace(0,1,len(CGF))

# Convert to real
s = np.real(s)
p = np.real(p)
CGF = np.real(CGF)
EE = np.real(EE)
nPart = np.real(nPart)
density = np.real(density)

# CGF Plot
if False:
    fig = plt.figure()
    fig.patch.set_facecolor('w')
    ax = fig.gca(projection='3d')
    ax.zaxis.set_rotate_label(False)
    sM,pM = np.meshgrid(s,p)
    surf = ax.plot_surface(sM,pM,CGF,cmap=colormap,linewidth=0,antialiased=False)
    plt.savefig('cgf_surf.pdf')

# Current Plot
if False:
    CGFt = CGF[::2,::2]
    st = s[::2]
    pt = p[::2]
    Current = (CGFt[:,1:]-CGFt[:,0:-1])/(st[0]-st[1])
    s_plt = st[0:-1]
    fig = plt.figure()
    fig.patch.set_facecolor('w')
    ax = fig.gca(projection='3d')
    sM,pM = np.meshgrid(s_plt,pt)
    surf = ax.plot_surface(sM,pM,Current,cmap=colormap,linewidth=0,antialiased=False)
    plt.savefig('current_surf.pdf')

# Susceptibility Image
if False:
    Current = (CGF[:,1:]-CGF[:,0:-1])/(s[0]-s[1])
    s_plt = s[0:-1]
    Susceptibility = (Current[:,1:]-Current[:,0:-1])/(s_plt[0]-s_plt[1])
    s_plt = s_plt[0:-1]
    fig = plt.figure()
    fig.patch.set_facecolor('w')
    ax = fig.gca()
    sM,pM = np.meshgrid(s_plt,p)
    surf = ax.pcolormesh(sM,pM,Susceptibility,cmap=colormap,linewidth=0,antialiased=False,vmin=0, vmax=10)
    ax.set_ylim(0,1)
    ax.set_xlim(-5,5)
    fig.colorbar(surf)
    plt.savefig('susc_surf.pdf')

# ee plot
if False:
    fig = plt.figure()
    fig.patch.set_facecolor('w')
    ax = fig.gca()
    sm,pm = np.meshgrid(s,p)
    surf = ax.pcolormesh(sm,pm,EE,cmap=colormap)
    ax.set_ylim(0,1)
    ax.set_xlim(-5,5)
    fig.colorbar(surf)
    plt.savefig('ee_surf.pdf')

###############################################
# Increase system size
filename = [None]*10
CGF = [None]*10
EE = [None]*10
nPart = [None]*10
density = [None]*10
plt_label = [None]*10
s = [None]*10
p = [None]*10
filename[0] = 'N10_data_p1s100.npz'
filename[1] = 'N20_data_p1s100.npz'
filename[2] = 'N30_data_p1s100.npz'
filename[3] = 'N40_data_p1s100.npz'
filename[4] = 'N50_data_p1s100.npz'
filename[5] = 'N60_data_p1s100.npz'
filename[6] = 'N70_data_p1s100.npz'
filename[7] = 'N80_data_p1s100.npz'
filename[8] = 'N90_data_p1s100.npz'
filename[9] = 'N100_data_p1s100.npz'
for i in range(len(filename)):
    npzfile = np.load(filename[i])
    CGF[i] = npzfile['CGF'][0,:]
    EE[i] = npzfile['EE'][0,:]
    nPart[i] = npzfile['nPart'][0,:]
    density[i] = npzfile['density'][0,:,:]
    s[i] = np.linspace(-5,5,len(CGF[i]))
    p[i] = np.array([0.2])
    # Convert to real
    s[i] = np.real(s[i])
    p[i] = np.real(p[i])
    CGF[i] = np.real(CGF[i])
    EE[i] = np.real(EE[i])
    nPart[i] = np.real(nPart[i])
    density[i] = np.real(density[i])
    plt_label[i] = '$N='+str((i+1)*10)+'$'

# Scaled CGF Graph
if False:
    fig = plt.figure()
    ax = fig.gca()
    del_inds = [31,68]
    pltCGF = [None]*len(CGF)
    plts = [None]*len(s)
    pltCGF[0] = np.delete(CGF[0],del_inds)
    pltCGF[1] = np.delete(CGF[1],del_inds)
    pltCGF[2] = np.delete(CGF[2],del_inds)
    pltCGF[3] = np.delete(CGF[3],del_inds)
    pltCGF[4] = np.delete(CGF[4],del_inds)
    pltCGF[5] = np.delete(CGF[5],del_inds)
    pltCGF[6] = np.delete(CGF[6],del_inds)
    plts[0] = np.delete(s[0],del_inds)
    plts[1] = np.delete(s[1],del_inds)
    plts[2] = np.delete(s[2],del_inds)
    plts[3] = np.delete(s[3],del_inds)
    plts[4] = np.delete(s[4],del_inds)
    plts[5] = np.delete(s[5],del_inds)
    plts[6] = np.delete(s[6],del_inds)
    surf = ax.plot(plts[1],pltCGF[1]/20.-pltCGF[6]/70.,label=plt_label[1],color=colormap(1./6.))
    surf = ax.plot(plts[2],pltCGF[2]/30.-pltCGF[6]/70.,label=plt_label[2],color=colormap(2./6.))
    surf = ax.plot(plts[3],pltCGF[3]/40.-pltCGF[6]/70.,label=plt_label[3],color=colormap(3./6.))
    surf = ax.plot(plts[4],pltCGF[4]/50.-pltCGF[6]/70.,label=plt_label[4],color=colormap(4./6.))
    surf = ax.plot(plts[5],pltCGF[5]/60.-pltCGF[6]/70.,label=plt_label[5],color=colormap(5./6.))
    surf = ax.plot(plts[6],pltCGF[6]/70.-pltCGF[6]/70.,label=plt_label[6],color=colormap(6./6.))
    #surf = ax.plot(plts[0],pltCGF[0]/10.-pltCGF[6]/70.,label=plt_label[0])
    ax2 = plt.axes([.6, .6, .25, .25], facecolor=(0.9,0.9,0.9))
    for i in range(len(filename)-3):
        surf = ax2.plot(plts[i],pltCGF[i],label=plt_label[i],color=colormap(i/6.))
    #ax.legend(loc=2)
    plt.savefig('cgf_1d.pdf')

# Scaled Current Graph
if False:
    fig = plt.figure()
    ax = fig.gca()
    ax2 = plt.axes([.6, .6, .25, .25], facecolor=(0.9,0.9,0.9))
    del_inds = [30,31,67,68]
    curr_large = (CGF[6][1:]-CGF[6][0:-1])/(s[6][0]-s[6][1])
    curr_large = np.delete(curr_large,del_inds)
    for i in range(1,len(filename)-3):
        Current = (CGF[i][1:]-CGF[i][0:-1])/(s[i][0]-s[i][1])
        s_plt = s[i][0:-1]
        Current = np.delete(Current,del_inds)
        s_plt = np.delete(s_plt,del_inds)
        surf = ax.plot(s_plt,Current/((i+1.)*10.+1.)-curr_large/((6.+1.)*10.+1.),label=plt_label[i],color=colormap(i/6.))
        surf = ax2.plot(s_plt,Current,color=colormap(i/6.))
    #ax.legend(loc=2)
    plt.savefig('current_1d.pdf')

# Susceptibility Graph
if False:
    fig = plt.figure()
    ax = fig.gca()
    del_inds = [29,30,31]
    for i in range(1,len(filename)-3):
        Current = (CGF[i][1:]-CGF[i][0:-1])/(s[i][0]-s[i][1])
        s_plt = s[i][0:-1]
        Susceptibility = (Current[1:]-Current[0:-1])/(s_plt[0]-s_plt[1])
        s_plt = s_plt[0:-1]
        Susceptibility = np.delete(Susceptibility,del_inds)
        s_plt = np.delete(s_plt,del_inds)
        surf = ax.plot(s_plt,Susceptibility/((i+1.)*10.+1.),label=plt_label[i],color=colormap(i/6.))
    ax.set_xlim(-2.5,1)
    ax.set_ylim(-0.5,2.5)
    #ax.legend(loc=1)
    plt.savefig('susc_1d.pdf')

# ee plot
if False:
    fig = plt.figure()
    ax = fig.gca()
    del_inds = [31,68]
    for i in range(1,len(filename)-3):
        surf = ax.plot(np.delete(s[i],del_inds),np.delete(EE[i],del_inds),label=plt_label[i],color=colormap(i/6.))
    #ax.legend(loc=1)
    plt.savefig('ee_1d.pdf')

# legend
if False:
    fig = plt.figure()
    ax = fig.gca()
    del_inds = [31,68]
    for i in range(1,len(filename)-3):
        surf = ax.plot(np.delete(s[i],del_inds),np.delete(EE[i],del_inds),label=plt_label[i],color=colormap(i/6.))
    ax.set_xlim(-100,-99)
    ax.set_ylim(-100,-99)
    ax.legend()
    plt.savefig('legend_1d.pdf')

###############################################
# 2 Lane ASEP Results
filename = [None]*4
CGF = [None]*4
EE = [None]*4
nPart = [None]*4
density = [None]*4
plt_label = [None]*4
s = [None]*4
p = [None]*4
L = 50.
filename[0] = 'N50_data_p1s100.npz'
filename[1] = '50x2_data_p1_s1pts_closed.npz'
filename[2] = '50x2_data_p1_s1pts_open.npz'
filename[3] = '50x2_data_p1_s1pts_periodic.npz'
for i in range(len(filename)):
    npzfile = np.load(filename[i])
    if i is 0:
        CGF[i] = npzfile['CGF'][0,:]
        EE[i] = npzfile['EE'][0,:]
        nPart[i] = npzfile['nPart'][0,:]
        density[i] = npzfile['density'][0,:,:]
        s[i] = np.linspace(-5,5,len(CGF[i]))
        p[i] = np.array([0.2])
        # Convert to real
        s[i] = np.real(s[i])
        p[i] = np.real(p[i])
        CGF[i] = np.real(CGF[i])
        EE[i] = np.real(EE[i])
        nPart[i] = np.real(nPart[i])
        density[i] = np.real(density[i])
    else:
        CGF[i] = npzfile['CGF'][0,:,0,0]
        EE[i] = npzfile['EE'][0,:,0,0]
        nPart[i] = npzfile['nPart'][0,:,0,0]
        density[i] = npzfile['density'][0,:,:,0,0]
        s[i] = np.linspace(-5,5,len(CGF[i]))
        p[i] = np.array([0.2])
        # Convert to real
        s[i] = np.real(s[i])
        p[i] = np.real(p[i])
        CGF[i] = np.real(CGF[i])
        EE[i] = np.real(EE[i])
        nPart[i] = np.real(nPart[i])
        density[i] = np.real(density[i])
plt_label[0] = '1D ASEP'
plt_label[1] = '2D Closed'
plt_label[2] = '2D Open'
plt_label[3] = '2D Periodic'

# Scaled CGF Graph
if True:
    fig = plt.figure()
    ax = fig.gca()
    ax2 = plt.axes([.25, .2, .25, .25], facecolor=(0.9,0.9,0.9))
    del_inds = [37]
    ax.plot(s[0],CGF[0]/L-CGF[0]/L,label=plt_label[0],color=colormap(0./6.))
    surf = ax.plot(np.delete(s[1],del_inds),np.delete(CGF[1],del_inds)/(2.*L)-np.delete(CGF[0],del_inds)/L,label=plt_label[1],color=colormap(2./6.))
    surf = ax.plot(s[2],CGF[2]/(2.*L)-CGF[0]/L,label=plt_label[2],color=colormap(4./6.))
    surf = ax.plot(np.delete(s[3],del_inds),np.delete(CGF[3],del_inds)/(2.*L)-np.delete(CGF[0],del_inds)/L,'.',label=plt_label[3],color=colormap(6./6.))
    surf = ax2.plot(s[0],CGF[0],color=colormap(0./6.))
    surf = ax2.plot(np.delete(s[1],del_inds),np.delete(CGF[1],del_inds),color=colormap(2./6.))
    surf = ax2.plot(s[2],CGF[2],color=colormap(4./6.))
    surf = ax2.plot(np.delete(s[3],del_inds),np.delete(CGF[3],del_inds),'.',color=colormap(6./6.))
    ax2.set_xlim(-2,1)
    ax2.set_ylim(-6,6)
    #ax.legend(loc=3)
    plt.savefig('cgf_2d.pdf')

# Scaled Current Graph
if False:
    fig = plt.figure()
    ax = fig.gca()
    del_inds = [36,37]
    ax2 = plt.axes([.6, .6, .25, .25], facecolor=(0.9,0.9,0.9))
    curr_large = (CGF[0][1:]-CGF[0][0:-1])/(s[0][0]-s[0][1])
    curr_large = np.delete(curr_large,del_inds)
    # Line 1
    i = 0
    Current = (CGF[i][1:]-CGF[i][0:-1])/(s[i][0]-s[i][1])
    s_plt = s[i][0:-1]
    Current = np.delete(Current,del_inds)
    s_plt = np.delete(s_plt,del_inds)
    surf = ax.plot(s_plt,Current/(50.)-curr_large/(50.),label=plt_label[i],color=colormap(0./6.))
    surf = ax2.plot(s_plt,Current,color=colormap(0./6.))
    # Line 1
    i = 1
    Current = (CGF[i][1:]-CGF[i][0:-1])/(s[i][0]-s[i][1])
    s_plt = s[i][0:-1]
    Current = np.delete(Current,del_inds)
    s_plt = np.delete(s_plt,del_inds)
    surf = ax.plot(s_plt,Current/(100.)-curr_large/(50.),label=plt_label[i],color=colormap(2./6.))
    surf = ax2.plot(s_plt,Current,color=colormap(2./6.))
    # Line 1
    i = 2
    Current = (CGF[i][1:]-CGF[i][0:-1])/(s[i][0]-s[i][1])
    s_plt = s[i][0:-1]
    Current = np.delete(Current,del_inds)
    s_plt = np.delete(s_plt,del_inds)
    surf = ax.plot(s_plt,Current/(100.)-curr_large/(50.),label=plt_label[i],color=colormap(4./6.))
    surf = ax2.plot(s_plt,Current,color=colormap(4./6.))
    # Line 1
    i = 3
    Current = (CGF[i][1:]-CGF[i][0:-1])/(s[i][0]-s[i][1])
    s_plt = s[i][0:-1]
    Current = np.delete(Current,del_inds)
    s_plt = np.delete(s_plt,del_inds)
    surf = ax.plot(s_plt,Current/(100.)-curr_large/(50.),'.',label=plt_label[i],color=colormap(6./6.))
    surf = ax2.plot(s_plt,Current,'.',color=colormap(6./6.))
    #ax.legend(loc=1)
    plt.savefig('current_2d.pdf')

# Susceptibility Graph
if False:
    fig = plt.figure()
    ax = fig.gca()
    del_inds = [35,37]#35,36,37]
    # Plot 
    i = 0
    Current = (CGF[i][1:]-CGF[i][0:-1])/(s[i][0]-s[i][1])
    s_plt = s[i][0:-1]
    Susceptibility = (Current[1:]-Current[0:-1])/(s_plt[0]-s_plt[1])
    s_plt = s_plt[0:-1]
    Susceptibility = np.delete(Susceptibility,del_inds)
    s_plt = np.delete(s_plt,del_inds)
    surf = ax.plot(s_plt,Susceptibility/(L),label=plt_label[i],color=colormap(0./6.))
    # Plot 
    i = 1
    Current = (CGF[i][1:]-CGF[i][0:-1])/(s[i][0]-s[i][1])
    s_plt = s[i][0:-1]
    Susceptibility = (Current[1:]-Current[0:-1])/(s_plt[0]-s_plt[1])
    s_plt = s_plt[0:-1]
    Susceptibility = np.delete(Susceptibility,del_inds)
    s_plt = np.delete(s_plt,del_inds)
    surf = ax.plot(s_plt,Susceptibility/(2.*L),label=plt_label[i],color=colormap(2./6.))
    # Plot 
    i = 2
    Current = (CGF[i][1:]-CGF[i][0:-1])/(s[i][0]-s[i][1])
    s_plt = s[i][0:-1]
    Susceptibility = (Current[1:]-Current[0:-1])/(s_plt[0]-s_plt[1])
    s_plt = s_plt[0:-1]
    Susceptibility = np.delete(Susceptibility,del_inds)
    s_plt = np.delete(s_plt,del_inds)
    surf = ax.plot(s_plt,Susceptibility/(2.*L),label=plt_label[i],color=colormap(4./6.))
    # Plot 
    i = 3
    Current = (CGF[i][1:]-CGF[i][0:-1])/(s[i][0]-s[i][1])
    s_plt = s[i][0:-1]
    Susceptibility = (Current[1:]-Current[0:-1])/(s_plt[0]-s_plt[1])
    s_plt = s_plt[0:-1]
    Susceptibility = np.delete(Susceptibility,del_inds)
    s_plt = np.delete(s_plt,del_inds)
    surf = ax.plot(s_plt,Susceptibility/(2.*L),'.',label=plt_label[i],color=colormap(6./6.))
    #ax.set_xlim(-2.5,1)
    ax.set_ylim(-0.5,10)
    #ax.legend(loc=1)
    plt.savefig('susc_2d.pdf')

# ee plot
if False:
    fig = plt.figure()
    ax = fig.gca()
    #Plot
    i = 0
    surf = ax.plot(s[i],EE[i],label=plt_label[i],color=colormap(0./6.))
    # Plot
    i = 1
    surf = ax.plot(s[i],EE[i],label=plt_label[i],color=colormap(2./6.))
    # Plot
    i = 2
    surf = ax.plot(s[i],EE[i],label=plt_label[i],color=colormap(4./6.))
    # Plot
    i = 3
    surf = ax.plot(s[i],EE[i],'.',label=plt_label[i],color=colormap(6./6.))
    #ax.legend(loc=1)
    plt.savefig('ee_2d.pdf')

if False:
    fig = plt.figure()
    ax = fig.gca()
    #Plot
    i = 0
    surf = ax.plot(s[i],EE[i],label=plt_label[i],color=colormap(0/6.))
    # Plot
    i = 1
    surf = ax.plot(s[i],EE[i],label=plt_label[i],color=colormap(2./6.))
    # Plot
    i = 2
    surf = ax.plot(s[i],EE[i],label=plt_label[i],color=colormap(4./6.))
    # Plot
    i = 3
    surf = ax.plot(s[i],EE[i],'.',label=plt_label[i],color=colormap(6./6.))
    ax.set_xlim(-100,-99)
    ax.set_ylim(-100,-99)
    ax.legend()
    plt.savefig('legend_2d.pdf')












# Density Profiles Plot
if False:
    len(p)
    print(density.shape)
    for p_ind in range(0,len(p),int(len(p)/20)):
        print('p = {}'.format(p_ind))
        # Trajectory 1
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #p_ind = 1
        for i in range(len(s)):
            if not i%int(len(s)/40):
                ax.plot(np.arange(len(density[p_ind,i,:])),s[i]*np.ones(len(density[p_ind,i,:])),density[p_ind,i,:],
                        'k-o',linewidth=1)
        #ax.set_xlabel('Site')
        #ax.set_ylabel('$\lambda$')
        #ax.set_zlabel('$\\rho$')
        ax.set_zlim(0,1)
    for s_ind in range(0,len(s),int(len(s)/20)):
        print('s = {}'.format(s_ind))
        # Trajectory 2
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #s_ind = 9
        for i in range(len(p)):
            if not i%int(len(p)/40):
                ax.plot(np.arange(len(density[i,s_ind,:])),p[i]*np.ones(len(density[i,s_ind,:])),density[i,s_ind,:],
                        'k-o',linewidth=1)
        #ax.set_xlabel('Site')
        #ax.set_ylabel('$p$')
        #ax.set_zlabel('$\\rho$')
        ax.set_zlim(0,1)


plt.show()
