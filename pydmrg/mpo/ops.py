import numpy as np

# A bunch of operators that are useful in other files
Sp = np.array([[0.,1.],
               [0.,0.]])

Sm = np.array([[0.,0.],
               [1.,0.]])

n = np.array([[0.,0.], 
              [0.,1.]])

v = np.array([[1.,0.],
              [0.,0.]])

I = np.array([[1.,0.],
              [0.,1.]])

z = np.array([[0.,0.],
              [0.,0.]])

Sx = 1./2.*np.array([[0.,1.],
                     [1.,0.]])

Sy = 1./(2.j)*np.array([[0.,1.],
                        [-1.,0.]])

Sz = 1./2.*np.array([[1.,0.],
                     [0.,-1.]])

X = 2.*Sx
Z = 2.*Sz
