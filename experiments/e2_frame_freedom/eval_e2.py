import os
import argparse
import pickle
import numpy as np
import jax
import jax.numpy as jnp

from src.envs.dmc_ood import make_dmc_env
from experiments.e2_frame_freedom.train_e2 import WorldModel, collect_random_dataset
from src.visualization.metrics import compute_orthogonality_deviation, orthogonal_procrustes
from src.visualization.plotter import plot_frame_freedom_vs_ood, save_plot

def evaluate_prediction_error(params, model, dataset):
    """Evaluates the 1-step latent prediction error normalized by std"""
    @jax.jit
    def compute_error(batch):
        z = model.apply({'params': params}, batch['states'], method=lambda m, s: m.encoder(s))
        z_next_pred = model.apply({'params': params}, z, batch['actions'], method=lambda m, z, a: m.dynamics(z, a))
        z_next_target = model.apply({'params': params}, batch['next_states'], method=lambda m, s: m.encoder(s))

        mse = jnp.mean((z_next_pred - z_next_target) ** 2, axis=-1)
        z_std = jnp.std(z_next_target, axis=0).mean()
        return jnp.mean(mse) / (z_std + 1e-6)

    batch = {k: jnp.array(v) for k, v in dataset.items()}
    return float(compute_error(batch))

def compute_diagnostics(params1, params2, model, dataset):
    r"""Computes orthogonality deviation \delta and collapse diagnostic (Effective Rank)"""
    states = jnp.array(dataset['states'])

    @jax.jit
    def get_latents(params, x):
        return model.apply({'params': params}, x, method=lambda m, s: m.encoder(s))

    Z1 = np.array(get_latents(params1, states))
    Z2 = np.array(get_latents(params2, states))

    def effective_rank(Z):
        Z_centered = Z - np.mean(Z, axis=0)
        cov = np.cov(Z_centered, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        p = eigenvalues / np.sum(eigenvalues)
        entropy = -np.sum(p * np.log(p))
        return np.exp(entropy)

    er_1 = effective_rank(Z1)
    er_2 = effective_rank(Z2)

    def whiten(Z):
        Z_centered = Z - np.mean(Z, axis=0)
        cov = np.cov(Z_centered, rowvar=False)
        U, S, Vt = np.linalg.svd(cov)
        W_zca = U @ np.diag(1.0 / np.sqrt(S + 1e-5)) @ U.T
        return Z_centered @ W_zca

    Z1_w = whiten(Z1)
    Z2_w = whiten(Z2)

    Q = orthogonal_procrustes(Z1_w, Z2_w)
    delta_ortho = np.linalg.norm(Z2_w - Z1_w @ Q.T, ord='fro') / np.sqrt(Z1_w.shape[0] * Z1_w.shape[1])

    W_lstsq, _, _, _ = np.linalg.lstsq(Z1_w, Z2_w, rcond=None)
    delta_linear = np.linalg.norm(Z2_w - Z1_w @ W_lstsq, ord='fro') / np.sqrt(Z1_w.shape[0] * Z1_w.shape[1])

    return delta_ortho, delta_linear, (er_1, er_2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='walker-walk')
    args = parser.parse_args()

    regularizers = ['full_rec', 'light_rec', 'vicreg', 'light_vicreg', 'sigreg']
    seeds = [42, 43]

    print("Collecting ID dataset...")
    env_id = make_dmc_env(*args.env.split("-"), mode="train", seed=99)
    id_data = collect_random_dataset(env_id, num_samples=5000)

    print("Collecting OOD dataset...")
    env_ood = make_dmc_env(*args.env.split("-"), mode="test_ood", seed=99)
    ood_data = collect_random_dataset(env_ood, num_samples=5000)

    obs_dim = id_data['states'].shape[1]
    model = WorldModel(obs_dim=obs_dim)

    results_delta = []
    results_ood_ratio = []
    labels = []

    for reg in regularizers:
        try:
            with open(f"outputs/checkpoints/e2/model_{args.env}_{reg}_seed{seeds[0]}.pkl", "rb") as f:
                params1 = pickle.load(f)
            with open(f"outputs/checkpoints/e2/model_{args.env}_{reg}_seed{seeds[1]}.pkl", "rb") as f:
                params2 = pickle.load(f)
        except FileNotFoundError:
            print(f"Skipping {reg} as checkpoints are missing.")
            continue

        print(f"--- Evaluating {reg} ---")

        delta_ortho, delta_linear, (er_1, er_2) = compute_diagnostics(params1, params2, model, id_data)

        id_err = evaluate_prediction_error(params1, model, id_data)
        ood_err = evaluate_prediction_error(params1, model, ood_data)
        ood_ratio = ood_err / id_err

        print(f"Eff Rank: S1={er_1:.1f}, S2={er_2:.1f} | Delta (Ortho): {delta_ortho:.4f}, Delta (Lin): {delta_linear:.4f} | ID Err: {id_err:.4f}, OOD Err: {ood_err:.4f}, Ratio: {ood_ratio:.4f}")

        results_delta.append(delta_ortho)
        results_ood_ratio.append(ood_ratio)
        labels.append(reg)

    if len(results_delta) > 0:
        fig = plot_frame_freedom_vs_ood(results_delta, results_ood_ratio, labels)
        save_plot(fig, f"e2_frame_freedom_{args.env}.png")
        print("Evaluation complete. Plot saved.")

if __name__ == "__main__":
    main()
