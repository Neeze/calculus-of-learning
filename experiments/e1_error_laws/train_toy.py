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
    """Creates initial `TrainState`."""
    dummy_state = jnp.ones((1, state_dim))
    dummy_action = jnp.ones((1, action_dim))
    variables = model.init(rng, dummy_state, dummy_action)
    params = variables['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch):
    """Trains for a single step."""
    def loss_fn(params):
        preds = state.apply_fn({'params': params}, batch['state'], batch['action'])
        # preds shape: (batch_size, k * d)
        # batch['target'] shape: (batch_size, k * d)
        loss = jnp.mean((preds - batch['target']) ** 2)
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step(state, batch, k=1):
    """Evaluates on a batch. Returns 1-step MSE specifically for matching."""
    preds = state.apply_fn({'params': state.params}, batch['state'], batch['action'])
    # Extract just the first step prediction (first d elements)
    d = batch['state'].shape[-1]
    pred_1step = preds[:, :d]
    target_1step = batch['target'][:, :d]
    loss_1step = jnp.mean((pred_1step - target_1step) ** 2)
    return loss_1step

def prepare_batches(data, k=1, batch_size=256):
    """
    Given data dictionary containing 'states' and 'actions', 'next_states',
    prepare batches of (s_t, a_{t:t+k-1}, s_{t+1:t+k})
    """
    states = data['states'] # (num_trajs, traj_length, d)
    actions = data['actions']
    num_trajs, traj_length, d = states.shape
    
    X_state, X_action, Y_target = [], [], []
    for t in range(traj_length - k + 1):
        # state at t
        X_state.append(states[:, t, :])
        # actions from t to t+k-1 flattened
        X_action.append(actions[:, t:t+k, :].reshape(num_trajs, -1))
        # target states from t+1 to t+k flattened
        target_seq = states[:, t+1:t+k+1, :] if t+k < traj_length else np.zeros((num_trajs, k, d))
        # Handle edge case where next_states provides the very last state
        if t+k == traj_length:
            target_seq = np.concatenate([states[:, t+1:t+k, :], data['next_states'][:, t+k-1:t+k, :]], axis=1)
        Y_target.append(target_seq.reshape(num_trajs, -1))
        
    X_state = np.concatenate(X_state, axis=0)
    X_action = np.concatenate(X_action, axis=0)
    Y_target = np.concatenate(Y_target, axis=0)
    
    # Shuffle
    indices = np.random.permutation(len(X_state))
    return {
        'state': X_state[indices],
        'action': X_action[indices],
        'target': Y_target[indices]
    }

def run_training_loop(model, train_data, val_data, state_dim, action_dim, k, epochs=100, lr=1e-3, target_eps=None):
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    
    state = create_train_state(init_rng, model, state_dim, action_dim, lr)
    
    train_batches = prepare_batches(train_data, k=k)
    val_batches = prepare_batches(val_data, k=k)
    
    batch_size = 256
    num_train_samples = len(train_batches['state'])
    num_batches = num_train_samples // batch_size
    
    history = []
    best_state = None
    best_diff = float('inf')
    
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
            
        # Validation
        val_loss_1step = eval_step(state, val_batches, k=k)
        
        # Save history for matched-eps logic
        history.append({
            'epoch': epoch,
            'val_loss_1step': float(val_loss_1step),
            'params': state.params
        })
        
        if target_eps is None: # Training M1, just keep the last/best
            best_state = state.params
            best_diff = val_loss_1step
        else:
            diff = abs(val_loss_1step - target_eps)
            if diff < best_diff:
                best_diff = diff
                best_state = state.params
                
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {epoch_loss/num_batches:.4f}, Val 1-step MSE: {val_loss_1step:.4f}")

    if target_eps is not None:
        print(f"Target ε: {target_eps:.4f}, Closest Val ε found: {history[-1]['val_loss_1step']:.4f} (Diff: {best_diff:.4f})")
        # In matched eps protocol, we select the one that is closest.
        # Check if it's within 2% band
        pct_diff = best_diff / target_eps
        print(f"Difference is {pct_diff*100:.2f}%")
        
    return best_state, history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='linear')
    parser.add_argument('--l_value', type=float, default=1.05)
    parser.add_argument('--seeds', type=int, default=1) # using 1 for toy script
    args = parser.parse_args()

    os.makedirs("outputs/checkpoints/e1", exist_ok=True)

    print(f"Generating data for L={args.l_value}...")
    train_data, val_data, test_data, A_matrix = generate_linear_dataset(target_L=args.l_value)
    
    # Save test data and A matrix for eval
    with open(f"outputs/checkpoints/e1/test_data_L{args.l_value}.pkl", "wb") as f:
        pickle.dump({'test_data': test_data, 'A_matrix': A_matrix}, f)

    d = 8
    
    # M1 (k=1)
    print("--- Training M1 (k=1) ---")
    m1 = create_m1_model(d=d)
    m1_params, m1_history = run_training_loop(m1, train_data, val_data, d, d, k=1, epochs=50)
    target_eps = min([h['val_loss_1step'] for h in m1_history]) # Use best validation error as target
    print(f"-> M1 Target ε (1-step MSE): {target_eps:.6f}")
    
    # M2 (k=4)
    print("--- Training M2 (k=4) ---")
    m2 = create_mk_model(d=d, k=4)
    m2_params, m2_history = run_training_loop(m2, train_data, val_data, d, d*4, k=4, epochs=50, target_eps=target_eps)
    
    # M3 (k=8)
    print("--- Training M3 (k=8) ---")
    m3 = create_mk_model(d=d, k=8)
    m3_params, m3_history = run_training_loop(m3, train_data, val_data, d, d*8, k=8, epochs=50, target_eps=target_eps)

    # Save models
    with open(f"outputs/checkpoints/e1/models_L{args.l_value}.pkl", "wb") as f:
        pickle.dump({
            'm1': m1_params,
            'm2': m2_params,
            'm3': m3_params
        }, f)
    
    print("Training complete and models saved.")

if __name__ == "__main__":
    main()
