import numpy as np

def pca_weights(cov, risk_dist=None, risk_target=1.0):
    # PCA Weights — AFML Chapter 2, Section 2.4.2, pages 35-36
    # (Snippet 2.1 in the book)
    #
    # Computes allocation weights ω for a portfolio of N instruments such that
    # risk is distributed across principal components according to a target
    # distribution R. Uses spectral decomposition of the covariance matrix V.
    #
    # Five-step derivation from pages 35-36:
    #
    # Step 1 — Spectral decomposition: VW = WΛ
    #   V = covariance matrix (NxN)
    #   W = eigenvectors (columns are principal components)
    #   Λ = diagonal matrix of eigenvalues (sorted descending)
    #
    # Step 2 — Portfolio risk: σ² = ω'Vω = β'Λβ
    #   β = W'ω  (projection of ω onto orthogonal eigenvector basis)
    #
    # Step 3 — Risk attribution per component:
    #   R_n = β²_n * Λ_n,n * σ^{-2}  (fraction of total risk from component n)
    #   Σ R_n = 1
    #
    # Step 4 — Allocation in orthogonal basis β:
    #   β_n = σ * sqrt(R_n / Λ_n,n)
    #   This is the allocation needed to achieve target risk distribution R
    #
    # Step 5 — Convert back to original basis:
    #   ω = W * β
    #
    # If risk_dist is None, all risk is allocated to the smallest eigenvalue
    # component (minimum variance portfolio).

    # Step 1: Spectral decomposition VW = WΛ
    # eVal = eigenvalues (diagonal of Λ), eVec = eigenvectors (columns of W)
    eVal, eVec = np.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors in descending order
    indices = eVal.argsort()[::-1]
    eVal, eVec = eVal[indices], eVec[:, indices]

    # Default: allocate all risk to the smallest eigenvalue component
    if risk_dist is None:
        risk_dist = np.zeros(cov.shape[0])
        risk_dist[-1] = 1.0

    # Step 4: β_n = σ * sqrt(R_n / Λ_n,n) — allocation in orthogonal basis
    # risk_target plays the role of σ (scales the overall portfolio risk)
    loads = risk_target * (risk_dist / eVal) ** 0.5

    # Step 5: ω = W * β — convert back to original instrument basis
    weights = np.dot(eVec, np.reshape(loads, (-1, 1)))

    return weights
