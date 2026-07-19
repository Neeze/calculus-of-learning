import os
import argparse
import pickle
import numpy as np
import jax
import jax.numpy as jnp

from src.envs.dmc_ood import make_dmc_env
from experiments.e2_frame_freedom.train_e2 import WorldModel, collect_random_dataset
from src.visualization.metrics import compute_orthogonality_deviation
from src.visualization.plotter import plot_frame_freedom_vs_ood, save_plot

def evaluate_prediction_error(params, model, dataset):
    """Evaluates the 1-step latent prediction error normalized by std"""
    @jax.jit
    def compute_error(batch):
        # We don't apply parameters here directly like typical train_state, we just pass params dict
        z = model.apply({'params': params}, batch['states'], method=lambda m, s: m.encoder(s))
        z_next_pred = model.apply({'params': params}, z, batch['actions'], method=lambda m, z, a: m.dynamics(z, a))
        z_next_target = model.apply({'params': params}, batch['next_states'], method=lambda m, s: m.encoder(s))
        
        mse = jnp.mean((z_next_pred - z_next_target) ** 2, axis=-1)
        z_std = jnp.std(z_next_target, axis=0).mean()
        return jnp.mean(mse) / (z_std + 1e-6)

    batch = {k: jnp.array(v) for k, v in dataset.items()}
    return float(compute_error(batch))

def compute_delta_between_seeds(params1, params2, model, dataset):
    """Computes orthogonality deviation \delta between the latent spaces of two seeds"""
    states = jnp.array(dataset['states'])
    
    @jax.jit
    def get_latents(params, x):
        return model.apply({'params': params}, x, method=lambda m, s: m.encoder(s))
        
    Z1 = get_latents(params1, states)
    Z2 = get_latents(params2, states)
    
    # Scale latents to roughly unit variance
    Z1 = Z1 / (jnp.std(Z1) + 1e-6)
    Z2 = Z2 / (jnp.std(Z2) + 1e-6)
    
    # Fit linear map W: Z1 -> Z2 using least squares
    # Z2 = Z1 @ W  => W = pinv(Z1) @ Z2
    W, _, _, _ = np.linalg.lstsq(np.array(Z1), np.array(Z2), rcond=None)
    
    delta = compute_orthogonality_deviation(W)
    return delta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='walker-walk')
    args = parser.parse_args()
    
    regularizers = ['full_rec', 'light_rec', 'vicreg', 'light_vicreg', 'jepa']
    seeds = [42, 43] # we assume we have 2 seeds trained for each regularizer
    
    # Collect ID and OOD evaluation data
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
        # 1. Compute delta
        delta = compute_delta_between_seeds(params1, params2, model, id_data)
        
        # 2. Compute OOD Ratio
        id_err = evaluate_prediction_error(params1, model, id_data)
        ood_err = evaluate_prediction_error(params1, model, ood_data)
        ood_ratio = ood_err / id_err
        
        print(f"Delta: {delta:.4f}, ID Err: {id_err:.4f}, OOD Err: {ood_err:.4f}, Ratio: {ood_ratio:.4f}")
        
        results_delta.append(delta)
        results_ood_ratio.append(ood_ratio)
        labels.append(reg)
        
    if len(results_delta) > 0:
        fig = plot_frame_freedom_vs_ood(results_delta, results_ood_ratio, labels)
        save_plot(fig, f"e2_frame_freedom_{args.env}.png")
        print("Evaluation complete. Plot saved.")

if __name__ == "__main__":
    main()
