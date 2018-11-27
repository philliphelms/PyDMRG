import numpy as np

def calcEE(N,wf,site=None):
    # Check that normalization is done correctly
    wf /= np.sqrt(np.sum(wf**2.))
    # Check that ordering is correct?
    # Reshape the wavefunction
    wf = np.reshape(wf,[int(2.**(site)),int(2.**(N-site))])
    # Decompose via SVD
    (U,S,V) = np.linalg.svd(wf,full_matrices=True)
    # Calculate Entanglement Entropy
    return -np.dot(S**2.,np.log2(S**2.))
