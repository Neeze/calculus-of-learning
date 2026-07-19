import numpy as np

def calculate_aic(n, mse, num_params):
    """
    Calculate Akaike Information Criterion for model selection
    AIC = 2k + n * ln(MSE)
    where k is the number of parameters, n is the number of observations
    """
    if mse <= 0:
        return np.inf
    return 2 * num_params + n * np.log(mse)

def compute_orthogonality_deviation(W):
    """
    Compute orthogonality deviation for E2
    δ = ||W^T W - I||_F / sqrt(d)
    """
    d = W.shape[1]
    I = np.eye(d)
    WT_W = np.dot(W.T, W)
    delta = np.linalg.norm(WT_W - I, ord='fro') / np.sqrt(d)
    return delta

def orthogonal_procrustes(A, B):
    """
    Find orthogonal matrix Q that minimizes ||B - QA||_F
    Used for E3 Plan Transfer
    """

    M = np.dot(B.T, A)
    U, _, Vt = np.linalg.svd(M)
    Q = np.dot(U, Vt)
    return Q

def calculate_return_ratio(transfer_return, baseline_return):
    """
    Return ratio = return(plan A via B) / return(B self plan)
    """
    return transfer_return / baseline_return
