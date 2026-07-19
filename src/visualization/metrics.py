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

def paired_bootstrap_test(a, b, n_boot=10000, seed=0):
    """
    Paired bootstrap for the one-sided hypothesis mean(a) > mean(b), e.g.
    h*(block predictor) vs h*(M1) across matched seeds.
    Returns (mean_diff, p_value) where p_value = P(bootstrap mean diff <= 0).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = a - b
    n = len(diffs)
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)
    ])
    mean_diff = diffs.mean()
    p_value = float(np.mean(boot_means <= 0.0))
    return mean_diff, p_value

def holm_correction(p_values):
    """
    Holm-Bonferroni step-down correction. Returns adjusted p-values in the
    same order as the input.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p_values[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted
