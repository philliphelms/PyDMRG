import pandas as pd
import numpy as np

fn = 'hamiltonian_3x3_s_0.xlsx'
xlfile = pd.ExcelFile(fn)
df = xlfile.parse("Sheet2")
H = np.zeros((512,512),dtype=np.float)
H[:,:] = df.as_matrix()[8:,1:]
#for i in range(512):
#    H[i,:] = np.array(df.as_matrix()[8+i,1:])
E_ed,_ = np.linalg.eig(H)
E_ed = np.sort(E_ed)[::-1]
print(E_ed)
for i in range(len(E_ed)):
    print(np.real(E_ed[i]))
print('Energy via Exact Diagonalization: {}'.format(np.sort(E_ed)[-1]))
