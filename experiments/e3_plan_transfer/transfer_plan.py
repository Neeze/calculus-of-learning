import os
import argparse
import pickle
import numpy as np
import jax
import jax.numpy as jnp

from src.envs.dmc_ood import make_dmc_env
from experiments.e2_frame_freedom.train_e2 import WorldModel, collect_random_dataset
from src.models.mappings import IdentityMap, OrthogonalProcrustesMap, AffineMap, SemanticAlignmentMap, MLPMap
from src.visualization.plotter import plot_trajectory_divergence, save_plot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='cartpole-swingup')
    args = parser.parse_args()
    
    seeds = [42, 43] # Models A and B
    
    # Load Models A and B
    try:
        with open(f"outputs/checkpoints/e2/model_{args.env}_jepa_seed{seeds[0]}.pkl", "rb") as f:
            params_A = pickle.load(f)
        with open(f"outputs/checkpoints/e2/model_{args.env}_jepa_seed{seeds[1]}.pkl", "rb") as f:
            params_B = pickle.load(f)
    except FileNotFoundError:
        print("Required checkpoints for E3 not found. Please run E2 training for 'jepa' with seeds 42 and 43 first.")
        # For toy demonstration purposes, we will exit gracefully.
        return
        
    print("Collecting Anchor dataset...")
    env = make_dmc_env(*args.env.split("-"), mode="train", seed=99)
    anchor_data = collect_random_dataset(env, num_samples=5000)
    
    obs_dim = anchor_data['states'].shape[1]
    model = WorldModel(obs_dim=obs_dim)
    
    @jax.jit
    def encode(params, obs):
        return model.apply({'params': params}, obs, method=lambda m, s: m.encoder(s))
        
    @jax.jit
    def unroll_dynamics(params, z0, actions):
        def scan_fn(z_t, a_t):
            z_next = model.apply({'params': params}, z_t, a_t, method=lambda m, z, a: m.dynamics(z, a))
            return z_next, z_next
        _, z_seq = jax.lax.scan(scan_fn, z0, actions)
        return jnp.concatenate([z0[None], z_seq], axis=0)

    states = jnp.array(anchor_data['states'])
    ZA = np.array(encode(params_A, states))
    ZB = np.array(encode(params_B, states))
    
    # Create mappings
    mappings = {
        'Identity': IdentityMap(),
        'Procrustes': OrthogonalProcrustesMap(),
        'Affine': AffineMap(),
        'Semantic': SemanticAlignmentMap(),
        'MLP': MLPMap(output_dim=ZB.shape[1])
    }
    
    print("Fitting mappings...")
    for name, mapping in mappings.items():
        print(f"Fitting {name}...")
        mapping.fit(ZA, ZB)
        
    # Generate Plan A
    print("Generating Plan A...")
    plan_data = collect_random_dataset(env, num_samples=50) # 50 steps horizon
    plan_obs = jnp.array(plan_data['states'])
    plan_actions = jnp.array(plan_data['actions'])
    
    # Get ground truth latent plan in A
    z_plan_A = encode(params_A, plan_obs)
    
    results = {}
    
    print("Transferring plan and unrolling in B...")
    for name, mapping in mappings.items():
        # Map the initial state
        z_A_0 = np.array(z_plan_A[0])
        z_B_0_mapped = mapping(z_A_0[None])[0]
        
        # Unroll using B's dynamics and A's actions
        z_B_unrolled = unroll_dynamics(params_B, jnp.array(z_B_0_mapped), plan_actions)
        
        # Ground truth mapped plan
        z_plan_A_mapped = mapping(np.array(z_plan_A))
        
        # Calculate divergence
        divergence = np.linalg.norm(np.array(z_B_unrolled[:-1]) - z_plan_A_mapped, axis=-1)
        results[name] = divergence
        
    # Plotting
    os.makedirs("outputs/plots", exist_ok=True)
    time_steps = np.arange(len(plan_actions))
    divergences = list(results.values())
    labels = list(results.keys())
    
    fig = plot_trajectory_divergence(time_steps, divergences, labels, title=f"Latent Trajectory Divergence ({args.env})")
    save_plot(fig, f"e3_plan_transfer_{args.env}.png")
    
    print("Evaluation complete. Results plotted.")

if __name__ == "__main__":
    main()
