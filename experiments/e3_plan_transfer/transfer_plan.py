import os
import argparse
import pickle
import numpy as np
import jax
import jax.numpy as jnp

from src.envs.dmc_ood import make_dmc_env
from experiments.e2_frame_freedom.train_e2 import WorldModel, collect_random_dataset
from src.models.mappings import IdentityMap, OrthogonalProcrustesMap, AffineMap, SemanticAlignmentMap, MLPMap, RelativeRepMap
from src.visualization.plotter import plot_trajectory_divergence, save_plot
import matplotlib.pyplot as plt

def evaluate_actions_in_env(env, physics_state, actions):
    with env.env.physics.reset_context():
        env.env.physics.set_state(physics_state)

    total_reward = 0.0
    for a in actions:
        _, r, _, _ = env.step(a)
        total_reward += r
    return total_reward

def cem_plan(model, params, z0, action_spec, horizon=50, num_iters=5, pop_size=200, num_elites=20, target_z_seq=None):
    """
    If target_z_seq is None, maximizes reward using model.reward_predictor.
    If target_z_seq is provided, minimizes MSE to it (tracking).
    """
    key = jax.random.PRNGKey(42)
    a_dim = action_spec.shape[0]

    mu = jnp.zeros((horizon, a_dim))
    std = jnp.ones((horizon, a_dim)) * (action_spec.maximum - action_spec.minimum) / 4.0

    @jax.jit
    def evaluate_trajectory(a_seq):
        def step(z, a):
            z_next = model.apply({'params': params}, z, a, method=lambda m, z, a: m.dynamics(z, a))
            reward = model.apply({'params': params}, z, a, method=lambda m, z, a: m.reward_predictor(z, a))
            return z_next, (z_next, reward)

        _, (z_seq, r_seq) = jax.lax.scan(step, z0, a_seq)

        if target_z_seq is not None:
            mse = jnp.mean((z_seq - target_z_seq)**2)
            return -mse
        else:
            return jnp.sum(r_seq)

    vmap_eval = jax.vmap(evaluate_trajectory)

    for _ in range(num_iters):
        key, subkey = jax.random.split(key)
        a_samples = mu + std * jax.random.normal(subkey, (pop_size, horizon, a_dim))
        a_samples = jnp.clip(a_samples, action_spec.minimum, action_spec.maximum)

        scores = vmap_eval(a_samples)

        elite_idx = jnp.argsort(scores)[-num_elites:]
        elites = a_samples[elite_idx]

        mu = jnp.mean(elites, axis=0)
        std = jnp.std(elites, axis=0)

    return np.array(mu)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='walker-walk')
    args = parser.parse_args()

    seeds = [42, 43]

    try:
        with open(f"outputs/checkpoints/e2/model_{args.env}_sigreg_seed{seeds[0]}.pkl", "rb") as f:
            params_A = pickle.load(f)
        with open(f"outputs/checkpoints/e2/model_{args.env}_sigreg_seed{seeds[1]}.pkl", "rb") as f:
            params_B = pickle.load(f)
    except FileNotFoundError:
        print("Required checkpoints for E3 not found. Please run E2 training for 'sigreg' with seeds 42 and 43 first.")
        return

    print("Collecting Anchor dataset...")
    env = make_dmc_env(*args.env.split("-"), mode="train", seed=99)
    anchor_data = collect_random_dataset(env, num_samples=5000)

    obs_dim = anchor_data['states'].shape[1]
    model = WorldModel(obs_dim=obs_dim)
    action_spec = env.env.action_spec()

    @jax.jit
    def encode(params, obs):
        return model.apply({'params': params}, obs, method=lambda m, s: m.encoder(s))

    @jax.jit
    def unroll_dynamics(params, z0, actions):
        def scan_fn(z_t, a_t):
            z_next = model.apply({'params': params}, z_t, a_t, method=lambda m, z, a: m.dynamics(z, a))
            return z_next, z_next
        _, z_seq = jax.lax.scan(scan_fn, z0, actions)
        return z_seq

    states = jnp.array(anchor_data['states'])
    ZA = np.array(encode(params_A, states))
    ZB = np.array(encode(params_B, states))

    print("Performing Sanity Check with Synthetic Rotation...")
    theta = np.pi / 4

    Q_true = np.eye(ZA.shape[1])
    Q_true[0:2, 0:2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    ZB_fake = np.dot(ZA, Q_true)

    sanity_map = OrthogonalProcrustesMap()
    sanity_map.fit(ZA, ZB_fake)
    q_err = np.linalg.norm(sanity_map.Q - Q_true)
    print(f"Sanity Check (Procrustes Q error): {q_err:.6f}")
    if q_err < 1e-4:
        print("-> Sanity Check PASSED!")
    else:
        print("-> Sanity Check FAILED!")

    mappings = {
        'Identity': IdentityMap(),
        'Procrustes': OrthogonalProcrustesMap(),
        'Affine': AffineMap(),
        'Relative': RelativeRepMap(),
        'Semantic': SemanticAlignmentMap(),
        'MLP': MLPMap(output_dim=ZB.shape[1])
    }

    print("Fitting mappings...")
    for name, mapping in mappings.items():
        print(f"Fitting {name}...")
        mapping.fit(ZA, ZB)

    print("Generating Plans...")
    env.reset()
    start_obs = env.reset()
    physics_state = env.env.physics.get_state().copy()

    z_A_0 = encode(params_A, jnp.array([start_obs]))[0]
    z_B_0 = encode(params_B, jnp.array([start_obs]))[0]

    horizon = 50

    actions_A = cem_plan(model, params_A, z_A_0, action_spec, horizon=horizon)
    return_A = evaluate_actions_in_env(env, physics_state, actions_A)

    actions_B = cem_plan(model, params_B, z_B_0, action_spec, horizon=horizon)
    return_B = evaluate_actions_in_env(env, physics_state, actions_B)

    actions_rand = np.random.uniform(action_spec.minimum, action_spec.maximum, size=(horizon, action_spec.shape[0]))
    return_rand = evaluate_actions_in_env(env, physics_state, actions_rand)

    print(f"Return A: {return_A:.2f}, Return B (Upper): {return_B:.2f}, Return Rand (Lower): {return_rand:.2f}")

    z_plan_A = unroll_dynamics(params_A, z_A_0, jnp.array(actions_A))

    results_div = {}
    results_return = {}

    print("Transferring plan and unrolling in B...")
    for name, mapping in mappings.items():
        z_plan_B_target = jnp.array(mapping(np.array(z_plan_A)))

        actions_mapped = cem_plan(model, params_B, z_B_0, action_spec, horizon=horizon, target_z_seq=z_plan_B_target)

        z_B_unrolled = np.array(unroll_dynamics(params_B, z_B_0, jnp.array(actions_mapped)))
        divergence = np.linalg.norm(z_B_unrolled - np.array(z_plan_B_target), axis=-1)
        results_div[name] = divergence

        ret = evaluate_actions_in_env(env, physics_state, actions_mapped)
        results_return[name] = ret

        ratio = (ret - return_rand) / (return_B - return_rand + 1e-8)
        print(f"[{name}] Return: {ret:.2f}, Ratio: {ratio*100:.2f}%")

    os.makedirs("outputs/plots", exist_ok=True)

    time_steps = np.arange(horizon)
    divergences = list(results_div.values())
    labels = list(results_div.keys())
    fig1 = plot_trajectory_divergence(time_steps, divergences, labels, title=f"Latent Trajectory Divergence ({args.env})")
    save_plot(fig1, f"e3_plan_transfer_div_{args.env}.png")

    fig2, ax = plt.subplots(figsize=(8, 6))
    ratios = [(results_return[name] - return_rand) / (return_B - return_rand + 1e-8) for name in labels]
    ax.bar(labels, ratios, color='skyblue')
    ax.axhline(1.0, color='r', linestyle='--', label='Upper Bound (B self-plan)')
    ax.axhline(0.0, color='k', linestyle='--', label='Lower Bound (Random)')
    ax.set_ylabel('Return Ratio')
    ax.set_title('Plan Transfer Return Ratio')
    ax.legend()
    save_plot(fig2, f"e3_plan_transfer_return_{args.env}.png")

    print("Evaluation complete. Results plotted.")

if __name__ == "__main__":
    main()
