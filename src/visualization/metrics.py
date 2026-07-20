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

def effective_rank(Z):
    """exp(entropy of covariance eigenvalue distribution) — collapse diagnostic
    (E2 spec §4a). Must be reported for every run."""
    Z_centered = Z - np.mean(Z, axis=0)
    cov = np.cov(Z_centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    p = eigenvalues / np.sum(eigenvalues)
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def linear_cka(Z1, Z2):
    """Linear CKA between two latent sets on the same inputs (E2 spec §4a)."""
    X = Z1 - Z1.mean(axis=0)
    Y = Z2 - Z2.mean(axis=0)
    hsic = np.linalg.norm(Y.T @ X, ord='fro') ** 2
    norm_x = np.linalg.norm(X.T @ X, ord='fro')
    norm_y = np.linalg.norm(Y.T @ Y, ord='fro')
    return float(hsic / (norm_x * norm_y + 1e-12))


def frame_freedom_metrics(Z1, Z2):
    """
    Frame-freedom measurements between two latent sets encoded from the SAME
    observations (E2 spec §4a).

    Normalization is center + global isotropic scale ONLY. Per-dimension
    whitening is deliberately NOT applied: it would erase exactly the variance
    structure that distinguishes the regularizer configs.

    Returns dict with:
      r2_linear   — R^2 of unconstrained least-squares map Z1 W ≈ Z2
      r2_ortho    — R^2 of the orthogonal Procrustes map
      delta_ortho — ||W̃ᵀW̃ − I||_F / √d for scale-normalized W̃ (deviation of
                    the best linear map from orthogonality)
      identity_dev— ||W̃ − I||_F / √d (recon configs should be near identity)
      cka         — linear CKA
    """
    def normalize(Z):
        Zc = Z - Z.mean(axis=0)
        scale = np.linalg.norm(Zc, ord='fro') / np.sqrt(Zc.shape[0] * Zc.shape[1])
        return Zc / (scale + 1e-12)

    Z1n, Z2n = normalize(Z1), normalize(Z2)
    d = Z1n.shape[1]
    denom = np.linalg.norm(Z2n, ord='fro') ** 2 + 1e-12

    W, _, _, _ = np.linalg.lstsq(Z1n, Z2n, rcond=None)
    r2_linear = 1.0 - np.linalg.norm(Z2n - Z1n @ W, ord='fro') ** 2 / denom

    Q = orthogonal_procrustes(Z1n, Z2n)  # minimizes ||Z2 - Z1 Qᵀ||
    r2_ortho = 1.0 - np.linalg.norm(Z2n - Z1n @ Q.T, ord='fro') ** 2 / denom

    W_tilde = W * np.sqrt(d) / (np.linalg.norm(W, ord='fro') + 1e-12)
    delta_ortho = np.linalg.norm(W_tilde.T @ W_tilde - np.eye(d), ord='fro') / np.sqrt(d)
    identity_dev = np.linalg.norm(W_tilde - np.eye(d), ord='fro') / np.sqrt(d)

    return {
        "r2_linear": float(r2_linear),
        "r2_ortho": float(r2_ortho),
        "delta_ortho": float(delta_ortho),
        "identity_dev": float(identity_dev),
        "cka": linear_cka(Z1, Z2),
    }


def ridge_probe_r2(Z_fit, Y_fit, Z_eval, Y_eval, alpha=1.0):
    """Ridge regression probe z -> ground-truth physics state, fit on ID,
    frozen, evaluated elsewhere (E2 spec §4b.3). Returns mean R^2 over targets."""
    Zf = Z_fit - Z_fit.mean(axis=0)
    Yf = Y_fit - Y_fit.mean(axis=0)
    d = Zf.shape[1]
    W = np.linalg.solve(Zf.T @ Zf + alpha * np.eye(d), Zf.T @ Yf)
    pred = (Z_eval - Z_fit.mean(axis=0)) @ W + Y_fit.mean(axis=0)
    ss_res = np.sum((Y_eval - pred) ** 2, axis=0)
    ss_tot = np.sum((Y_eval - Y_eval.mean(axis=0)) ** 2, axis=0) + 1e-12
    return float(np.mean(1.0 - ss_res / ss_tot))


def spearman_correlation(x, y):
    """Spearman rho with a permutation p-value (no scipy dependency)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rho = np.corrcoef(rx, ry)[0, 1]
    rng = np.random.default_rng(0)
    n_perm = 10000
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(ry)
        if abs(np.corrcoef(rx, perm)[0, 1]) >= abs(rho):
            count += 1
    return float(rho), float(count / n_perm)


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
