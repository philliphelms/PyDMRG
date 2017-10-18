import numpy as np
import matplotlib.pyplot as plt

# Number of Sites
L = 8
# SEP Parameters
alpha = 0.35 # Enter at left
beta = 0  # Enter at right
delta = 2/3   # Leave at right
gamma = 0   # Leave at left
p = 1     # Hop right
q = 0       # Hop left
s = -1

# Other useful vars
s_p = np.array([[0,1],
                [0,0]])
s_m = np.array([[0,0],
                [1,0]])
n = np.array([[0,0],
              [0,1]])
v = np.array([[1,0],
              [0,0]])

# Function for calculating occupation from 
def int2stringVec(i,L):
    occString = "{0:b}".format(i)
    if len(occString) < L:
        occString = "0"*(L-len(occString)) + occString
    occVec = np.zeros(L)
    for j in range(len(occString)):
        occVec[j] = int(occString[j])
    return occVec

print('='*50+'\nCreating Hamiltonian\n'+'='*50)
H = np.zeros([2**L,2**L])
for i in range(2**L):
    print('Percent Complete: {}'.format((i*(2**L))/((2**L)*(2**L))*100))
    occVec_i = int2stringVec(i,L)
    for j in range(2**L):
        occVec_j = int2stringVec(j,L)
        # Enter at Left Site:
        H[i,j] += alpha*(np.exp(-s)*s_m[occVec_i[0],occVec_j[0]]-v[occVec_i[0],occVec_j[0]])
        # Exit at Left Site        
        H[i,j] += 0
        # Enter at Right Site
        H[i,j] += delta*(np.exp(-s)*s_p[occVec_i[-1],occVec_j[-1]]-n[occVec_i[-1],occVec_j[-1]])        
        # Exit at Right Site
        H[i,j] += 0
        # Center Sites
        for k in range(L):
            # Hopping Right
            H[i,j] += p*(np.exp(-s)*s_p[occVec_i[k],occVec_j[k]]*s_m[occVec_i[k],occVec_j[k]]-n[occVec_i[k],occVec_j[k]]*v[occVec_i[k],occVec_j[k]])
            # Hopping Left
            H[i,j] += q*(np.exp(-s)*s_m[occVec_i[k],occVec_j[k]]*s_p[occVec_i[k],occVec_j[k]]-v[occVec_i[k],occVec_j[k]]*n[occVec_i[k],occVec_j[k]])
        print('Calculating <{}|H|{}> = {}'.format(occVec_i,occVec_j,H[i,j]))

# Calculate Eigenvalue
print(H)
w,v = np.linalg.eig(H)
pick_ind = 0#2**L-1
ind = np.argsort(w)[pick_ind]
e = w[ind]
print(np.sort(w)/(L+1))
print(e/(L+1))
v = v[:,ind]

# Calculate average occupancy:
avgOcc = np.zeros(L,dtype=np.complex128)
for i in range(2**L):
    #print(v[i]**2)
    #print(int2stringVec(i,L))
    #print(v[i]**2*int2stringVec(i,L))
    avgOcc += v[i]**2*int2stringVec(i,L)

plt.figure()
plt.plot(range(L),avgOcc,'r-')
plt.show()






