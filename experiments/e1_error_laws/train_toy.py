import os
import argparse
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import numpy as np
import pickle

from src.envs.linear import generate_linear_dataset
from src.models.mlp_predictors import create_m1_model, create_mk_model

def create_train_state(rng, model, state_dim, action_dim, k, learning_rate):
    dummy_state = jnp.ones((1, state_dim))
    if k == 1:
        dummy_action = jnp.ones((1, action_dim))
    else:
        dummy_action = jnp.ones((1, k, action_dim))
    variables = model.init(rng, dummy_state, dummy_action)
    params = variables['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        pred = state.apply_fn({'params': params}, batch['state'], batch['action'])
        loss = jnp.mean((pred - batch['target']) ** 2)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def eval_onestep_error(state, batch, state_std, is_block):
    """
    One-step matching error epsilon. For block predictors, this is the
    error of the FIRST step of the predicted block (see E1 spec §3.4), so
    it is directly comparable to M1's one-step error.
    """
    pred = state.apply_fn({'params': state.params}, batch['state'], batch['action'])
    if is_block:
        pred_1 = pred[..., 0, :]
        target_1 = batch['target'][..., 0, :]
    else:
        pred_1 = pred
        target_1 = batch['target']
    return float(jnp.mean(jnp.linalg.norm(pred_1 - target_1, axis=-1) / (state_std + 1e-8)))

def prepare_batches_onestep(data, rng):
    """One-step (state, action) -> next_state pairs, for M1."""
    states = data['states']
    actions = data['actions']
    next_states = data['next_states']
    num_trajs, traj_length, d = states.shape

    X_state = states.reshape(-1, d)
    X_action = actions.reshape(-1, actions.shape[-1])
    Y_target = next_states.reshape(-1, d)

    indices = rng.permutation(len(X_state))
    return {
        'state': X_state[indices],
        'action': X_action[indices],
        'target': Y_target[indices],
    }

def prepare_batches_block(data, k, rng):
    """
    (state_t, actions_{t..t+k-1}) -> block target states_{t+1..t+k}, for
    block predictors M2/M3. Only uses windows fully contained in the
    trajectory (no zero-padded partial blocks).
    """
    states = data['states']
    actions = data['actions']
    next_states = data['next_states']
    num_trajs, traj_length, d = states.shape

    X_state, X_action, Y_target = [], [], []
    for t in range(traj_length - k + 1):
        X_state.append(states[:, t, :])
        X_action.append(actions[:, t:t + k, :])
        Y_target.append(next_states[:, t:t + k, :])

    X_state = np.concatenate(X_state, axis=0)
    X_action = np.stack(X_action, axis=1).reshape(-1, k, actions.shape[-1])
    Y_target = np.stack(Y_target, axis=1).reshape(-1, k, d)

    indices = rng.permutation(len(X_state))
    return {
        'state': X_state[indices],
        'action': X_action[indices],
        'target': Y_target[indices],
    }

def run_training_loop(
    model, train_data, val_data, test_data, state_std, state_dim, action_dim, k,
    epochs=150, lr=1e-3, target_eps=None, seed=0,
):
    is_block = k > 1
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    np_rng = np.random.default_rng(seed)

    state = create_train_state(init_rng, model, state_dim, action_dim, k, lr)

    prep_fn = prepare_batches_block if is_block else prepare_batches_onestep
    prep_args = (train_data, k, np_rng) if is_block else (train_data, np_rng)
    train_batches = prep_fn(*prep_args)

    val_prep_args = (val_data, k, np_rng) if is_block else (val_data, np_rng)
    val_batches = prep_fn(*val_prep_args)

    batch_size = 256
    num_train_samples = len(train_batches['state'])
    num_batches = num_train_samples // batch_size

    history = []
    best_state = None
    best_diff = float('inf')
    best_val = None

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch = {
                'state': train_batches['state'][start_idx:end_idx],
                'action': train_batches['action'][start_idx:end_idx],
                'target': train_batches['target'][start_idx:end_idx],
            }
            state, loss = train_step(state, batch)
            epoch_loss += loss

        val_loss_1step = eval_onestep_error(state, val_batches, state_std, is_block)

        history.append({
            'epoch': epoch,
            'val_loss_1step': val_loss_1step,
        })

        if target_eps is None:
            best_state = state.params
            best_diff = val_loss_1step
            best_val = best_diff
        else:
            diff = abs(val_loss_1step - target_eps)
            if diff < best_diff:
                best_diff = diff
                best_state = state.params
                best_val = val_loss_1step

        if epoch % 30 == 0:
            print(f"Epoch {epoch}, Train Loss: {epoch_loss/num_batches:.4f}, Val E(1): {val_loss_1step:.4f}")

    if target_eps is not None:
        print(f"Target eps: {target_eps:.6f}, Closest Val E(1) found: {best_val:.6f} (Diff: {best_diff:.6f})")
        pct_diff = best_diff / target_eps
        print(f"Difference is {pct_diff*100:.2f}%")

    # Final params for reporting: use the best (matched) params found above.
    final_state = state.replace(params=best_state)
    test_prep_args = (test_data, k, np_rng) if is_block else (test_data, np_rng)
    test_batches = prep_fn(*test_prep_args)
    eps_val = eval_onestep_error(final_state, val_batches, state_std, is_block)
    eps_test = eval_onestep_error(final_state, test_batches, state_std, is_block)

    return best_state, history, eps_val, eps_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--l_values', type=float, nargs='+', default=[0.8, 0.95, 1.05, 1.2])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--epochs', type=int, default=150)
    args = parser.parse_args()

    os.makedirs("outputs/checkpoints/e1", exist_ok=True)
    d = 8

    matched_eps_rows = []

    for L in args.l_values:
        for seed in args.seeds:
            print(f"\n========== L={L}, Seed={seed} ==========")
            print(f"Generating data...")
            train_data, val_data, test_data, A_matrix, B_matrix = generate_linear_dataset(
                target_L=L, seed=seed,
            )

            state_std = np.std(train_data['states'])

            with open(f"outputs/checkpoints/e1/test_data_L{L}_seed{seed}.pkl", "wb") as f:
                pickle.dump({
                    'test_data': test_data,
                    'A_matrix': A_matrix,
                    'B_matrix': B_matrix,
                    'state_std': state_std,
                }, f)

            print("--- Training M1 (one-step) ---")
            m1 = create_m1_model(d=d)
            m1_params, m1_history, eps_val_m1, eps_test_m1 = run_training_loop(
                m1, train_data, val_data, test_data, state_std, d, d, k=1,
                epochs=args.epochs, seed=seed,
            )
            target_eps = m1_history[-1]['val_loss_1step']
            print(f"-> M1 Target eps (Val E(1)): {target_eps:.6f}")
            matched_eps_rows.append((L, seed, 'M1', eps_val_m1, eps_test_m1, 0.0))

            print("--- Training M2 (block k=4) ---")
            m2 = create_mk_model(d=d, k=4)
            m2_params, _, eps_val_m2, eps_test_m2 = run_training_loop(
                m2, train_data, val_data, test_data, state_std, d, d, k=4,
                epochs=args.epochs, target_eps=target_eps, seed=seed,
            )
            pct_m2 = 100.0 * (eps_val_m2 - eps_val_m1) / eps_val_m1
            matched_eps_rows.append((L, seed, 'M2', eps_val_m2, eps_test_m2, pct_m2))

            print("--- Training M3 (block k=8) ---")
            m3 = create_mk_model(d=d, k=8)
            m3_params, _, eps_val_m3, eps_test_m3 = run_training_loop(
                m3, train_data, val_data, test_data, state_std, d, d, k=8,
                epochs=args.epochs, target_eps=target_eps, seed=seed,
            )
            pct_m3 = 100.0 * (eps_val_m3 - eps_val_m1) / eps_val_m1
            matched_eps_rows.append((L, seed, 'M3', eps_val_m3, eps_test_m3, pct_m3))

            with open(f"outputs/checkpoints/e1/models_L{L}_seed{seed}.pkl", "wb") as f:
                pickle.dump({
                    'm1': m1_params,
                    'm2': m2_params,
                    'm3': m3_params,
                }, f)

    print("\n========== Matched-epsilon table (soul of E1) ==========")
    print(f"{'L':>6} {'seed':>6} {'model':>6} {'eps_val':>10} {'eps_test':>10} {'%diff vs M1':>12}")
    for L, seed, name, eps_val, eps_test, pct in matched_eps_rows:
        print(f"{L:6.2f} {seed:6d} {name:>6} {eps_val:10.6f} {eps_test:10.6f} {pct:11.2f}%")

    os.makedirs("outputs/plots", exist_ok=True)
    with open("outputs/plots/e1_matched_eps_table.csv", "w") as f:
        f.write("L,seed,model,eps_val,eps_test,pct_diff_vs_m1\n")
        for row in matched_eps_rows:
            f.write(",".join(str(x) for x in row) + "\n")

    print("All trainings complete!")

if __name__ == "__main__":
    main()
