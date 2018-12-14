import numpy as np

def return_mpo(N,hamParams):
    # Unpack Ham Params ################################
    a = hamParams[0]
    g = hamParams[1]
    p = hamParams[2]
    q = hamParams[3]
    b = hamParams[4]
    d = hamParams[5]
    s = hamParams[6]
    exp_a = a*np.exp(s)
    exp_g = g*np.exp(-s)
    exp_p = p*np.exp(s)
    exp_q = q*np.exp(-s)
    exp_b = b*np.exp(s)
    exp_d = d*np.exp(-s)
    # Create MPO #######################################
    Sp = np.array([[0,1],[0,0]])
    Sm = np.array([[0,0],[1,0]])
    n = np.array([[0,0],[0,1]])
    v = np.array([[1,0],[0,0]])
    I = np.array([[1,0],[0,1]])
    z = np.array([[0,0],[0,0]])
    W = []
    W.append(np.array([[exp_a*Sm-a*v+exp_g*Sp-g*n, Sp, -n, Sm, -v, I]]))
    for i in range(N-2):
        W.append(np.array([[I       ,  z,  z,  z,  z,  z],
                           [exp_p*Sm,  z,  z,  z,  z,  z],
                           [p*v     ,  z,  z,  z,  z,  z],
                           [exp_q*Sp,  z,  z,  z,  z,  z],
                           [q*n     ,  z,  z,  z,  z,  z],
                           [z       , Sp, -n, Sm, -v,  I]]))
    W.append(np.array([[I],
                     [exp_p*Sm],
                     [p*v],
                     [exp_q*Sp],
                     [q*n],
                     [exp_d*Sm-d*v+exp_b*Sp-b*n]]))
    return W
