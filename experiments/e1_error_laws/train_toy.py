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

def create_train_state(rng, model, state_dim, action_dim, learning_rate):
    dummy_state = jnp.ones((1, state_dim))
    dummy_action = jnp.ones((1, action_dim))
    variables = model.init(rng, dummy_state, dummy_action)
    params = variables['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        def unroll(curr_state, actions):
            def step(c_state, act):
                n_state = state.apply_fn({'params': params}, c_state, act)
                return n_state, n_state
            _, preds = jax.lax.scan(step, curr_state, actions)
            return preds

        batch_unroll = jax.vmap(unroll)
        preds = batch_unroll(batch['state'], batch['action'])
        loss = jnp.mean((preds - batch['target']) ** 2)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step(state, batch, state_std):
    pred_1step = state.apply_fn({'params': state.params}, batch['state'], batch['action'][:, 0, :])
    target_1step = batch['target'][:, 0, :]

    loss_1step = jnp.mean(jnp.linalg.norm(pred_1step - target_1step, axis=-1) / (state_std + 1e-8))
    return loss_1step

def prepare_batches(data, k=1):
    states = data['states']
    actions = data['actions']
    num_trajs, traj_length, d = states.shape

    X_state, X_action, Y_target = [], [], []
    for t in range(traj_length - k + 1):
        X_state.append(states[:, t, :])
        X_action.append(actions[:, t:t+k, :])

        target_seq = states[:, t+1:t+k+1, :] if t+k < traj_length else np.zeros((num_trajs, k, d))
        if t+k == traj_length:
            target_seq = np.concatenate([states[:, t+1:t+k, :], data['next_states'][:, t+k-1:t+k, :]], axis=1)
        Y_target.append(target_seq)

    X_state = np.concatenate(X_state, axis=0)
    X_action = np.concatenate(X_action, axis=0)
    Y_target = np.concatenate(Y_target, axis=0)

    indices = np.random.permutation(len(X_state))
    return {
        'state': X_state[indices],
        'action': X_action[indices],
        'target': Y_target[indices]
    }

def run_training_loop(model, train_data, test_data, state_std, state_dim, action_dim, k, epochs=150, lr=1e-3, target_eps=None):
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(init_rng, model, state_dim, action_dim, lr)

    train_batches = prepare_batches(train_data, k=k)

    test_batches = prepare_batches(test_data, k=k)

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
                'target': train_batches['target'][start_idx:end_idx]
            }
            state, loss = train_step(state, batch)
            epoch_loss += loss

        val_loss_1step = eval_step(state, test_batches, state_std)

        history.append({
            'epoch': epoch,
            'val_loss_1step': float(val_loss_1step),
        })

        if target_eps is None:
            best_state = state.params
            best_diff = float(val_loss_1step)
            best_val = best_diff
        else:
            diff = abs(val_loss_1step - target_eps)
            if diff < best_diff:
                best_diff = diff
                best_state = state.params
                best_val = float(val_loss_1step)

        if epoch % 30 == 0:
            print(f"Epoch {epoch}, Train Loss: {epoch_loss/num_batches:.4f}, Test E(1): {val_loss_1step:.4f}")

    if target_eps is not None:
        print(f"Target ε: {target_eps:.6f}, Closest Test E(1) found: {best_val:.6f} (Diff: {best_diff:.6f})")
        pct_diff = best_diff / target_eps
        print(f"Difference is {pct_diff*100:.2f}%")

    return best_state, history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--l_values', type=float, nargs='+', default=[0.8, 0.95, 1.05, 1.2])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    args = parser.parse_args()

    os.makedirs("outputs/checkpoints/e1", exist_ok=True)
    d = 8

    for L in args.l_values:
        for seed in args.seeds:
            print(f"\n========== L={L}, Seed={seed} ==========")
            print(f"Generating data...")
            train_data, val_data, test_data, A_matrix = generate_linear_dataset(target_L=L, seed=seed)

            state_std = np.std(train_data['states'])

            with open(f"outputs/checkpoints/e1/test_data_L{L}_seed{seed}.pkl", "wb") as f:
                pickle.dump({'test_data': test_data, 'A_matrix': A_matrix, 'state_std': state_std}, f)

            print("--- Training M1 (k=1) ---")
            m1 = create_m1_model(d=d)
            m1_params, m1_history = run_training_loop(m1, train_data, test_data, state_std, d, d, k=1, epochs=150)
            target_eps = m1_history[-1]['val_loss_1step']
            print(f"-> M1 Target ε (Test E(1)): {target_eps:.6f}")

            print("--- Training M2 (k=4) ---")
            m2 = create_mk_model(d=d, k=4)
            m2_params, _ = run_training_loop(m2, train_data, test_data, state_std, d, d, k=4, epochs=150, target_eps=target_eps)

            print("--- Training M3 (k=8) ---")
            m3 = create_mk_model(d=d, k=8)
            m3_params, _ = run_training_loop(m3, train_data, test_data, state_std, d, d, k=8, epochs=150, target_eps=target_eps)

            with open(f"outputs/checkpoints/e1/models_L{L}_seed{seed}.pkl", "wb") as f:
                pickle.dump({
                    'm1': m1_params,
                    'm2': m2_params,
                    'm3': m3_params
                }, f)

    print("All trainings complete!")

if __name__ == "__main__":
    main()
