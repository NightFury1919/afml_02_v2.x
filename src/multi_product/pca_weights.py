import numpy as np

def pca_weights(cov, risk_dist=None, risk_target=1.0):
    # Spectral decomposition: VW = WΛ
    # eVal = eigenvalues (Λ diagonal), eVec = eigenvectors (W columns)
    eVal, eVec = np.linalg.eigh(cov)

    # Sort eigenvalues in descending order
    indices = eVal.argsort()[::-1]
    eVal, eVec = eVal[indices], eVec[:, indices]

    # If no risk distribution provided, allocate all risk to smallest eigenvalue component
    if risk_dist is None:
        risk_dist = np.zeros(cov.shape[0])
        risk_dist[-1] = 1.0

    # β = σ * sqrt(R_n / Λ_n,n) — allocation in orthogonal basis
    loads = risk_target * (risk_dist / eVal) ** 0.5

    # ω = W * β — allocation in original basis
    weights = np.dot(eVec, np.reshape(loads, (-1, 1)))

    return weights