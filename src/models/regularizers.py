import jax
import jax.numpy as jnp

def sigreg_loss(embeddings: jnp.ndarray, rng: jax.random.PRNGKey, num_slices: int = 256) -> jnp.ndarray:
    """
    Sketched Isotropic Gaussian Regularization (SIGReg)
    Implementation based on LeJEPA paper (Epps-Pulley test).

    Args:
        embeddings: (N, K) tensor of embeddings.
        rng: JAX random key for projection directions.
        num_slices: Number of random projections (M).

    Returns:
        Scalar loss value.
    """
    N, K = embeddings.shape

    A = jax.random.normal(rng, (K, num_slices))
    A = A / jnp.linalg.norm(A, axis=0, keepdims=True)

    t = jnp.linspace(-5, 5, 17)

    exp_f = jnp.exp(-0.5 * (t ** 2))

    x_t = jnp.expand_dims(jnp.dot(embeddings, A), axis=-1) * t

    ecf = jnp.mean(jnp.exp(1j * x_t), axis=0)

    diff = ecf - exp_f
    err = (jnp.real(diff)**2 + jnp.imag(diff)**2) * exp_f

    dt = t[1] - t[0]

    integral = jnp.sum((err[:, :-1] + err[:, 1:]) / 2.0 * dt, axis=-1)

    loss = jnp.mean(integral) * N
    return loss

def vicreg_loss(embeddings: jnp.ndarray, var_weight: float = 25.0, cov_weight: float = 1.0) -> jnp.ndarray:
    """
    VICReg regularization: Variance + Covariance
    (Invariance is handled separately by the prediction loss)
    """
    N, K = embeddings.shape

    std_x = jnp.sqrt(jnp.var(embeddings, axis=0) + 1e-04)
    std_loss = jnp.mean(jnp.maximum(0, 1.0 - std_x))

    x = embeddings - jnp.mean(embeddings, axis=0)
    cov_x = (jnp.dot(x.T, x)) / (N - 1)
    cov_loss = (jnp.sum(cov_x ** 2) - jnp.sum(jnp.diag(cov_x) ** 2)) / K

    return var_weight * std_loss + cov_weight * cov_loss

def reconstruction_loss(pred_pixels: jnp.ndarray, target_pixels: jnp.ndarray) -> jnp.ndarray:
    """
    Simple MSE reconstruction loss.
    """
    return jnp.mean((pred_pixels - target_pixels) ** 2)
