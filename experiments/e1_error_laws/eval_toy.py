import os
import argparse
import pickle
import numpy as np
import jax.numpy as jnp
from scipy.optimize import curve_fit

from src.models.mlp_predictors import create_m1_model, create_mk_model
from src.visualization.metrics import calculate_aic
from src.visualization.plotter import plot_error_law, save_plot

def rollout_m1(params, model, start_state, actions):
    """Rollout M1 autoregressively."""
    H = actions.shape[0]
    preds = []
    curr_state = start_state
    for t in range(H):
        curr_state = model.apply({'params': params}, curr_state, actions[t])
        preds.append(curr_state)
    return np.array(preds)

def rollout_mk(params, model, start_state, actions, k):
    """Rollout Mk by jumping k steps at a time."""
    H = actions.shape[0]
    preds = []
    curr_state = start_state
    
    for t in range(0, H, k):
        # We need k actions. If not enough, pad with zeros
        a_chunk = actions[t:t+k]
        actual_k = len(a_chunk)
        if actual_k < k:
            a_chunk = np.pad(a_chunk, ((0, k - actual_k), (0, 0)))
        
        a_flat = a_chunk.reshape(-1)
        pred_k = model.apply({'params': params}, curr_state, a_flat)
        pred_k = pred_k.reshape(k, -1)
        
        preds.extend(pred_k[:actual_k])
        curr_state = pred_k[-1]
        
    return np.array(preds)

def integrator_law(h, eps, L):
    L_safe = np.where(np.abs(L - 1.0) < 1e-6, 1.0 + 1e-6, L)
    return eps * (np.power(L_safe, h) - 1.0) / (L_safe - 1.0)

def linear_law(h, c):
    return c * h

def power_law(h, c, p):
    return c * (h ** p)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--l_value', type=float, default=1.05)
    args = parser.parse_args()

    # Load data
    with open(f"outputs/checkpoints/e1/test_data_L{args.l_value}.pkl", "rb") as f:
        data = pickle.load(f)
        test_data = data['test_data']
        A_matrix = data['A_matrix']
        
    actual_L = np.linalg.norm(A_matrix, ord=2)
    print(f"Actual measured L: {actual_L:.4f}")

    # Load models
    with open(f"outputs/checkpoints/e1/models_L{args.l_value}.pkl", "rb") as f:
        models_params = pickle.load(f)

    d = 8
    m1 = create_m1_model(d=d)
    m2 = create_mk_model(d=d, k=4)
    m3 = create_mk_model(d=d, k=8)

    states = test_data['states'] # (num_trajs, H, d)
    actions = test_data['actions']
    num_trajs, H, _ = states.shape
    
    # We will compute error for h=1..50 to avoid saturation too early
    eval_H = min(50, H - 1)
    
    errors_m1, errors_m2, errors_m3 = [], [], []
    
    state_std = np.std(states) # normalization factor

    print(f"Rolling out {num_trajs} trajectories for {eval_H} steps...")
    for i in range(num_trajs):
        s0 = states[i, 0]
        a_seq = actions[i, :eval_H]
        target_seq = test_data['next_states'][i, :eval_H]
        
        preds_m1 = rollout_m1(models_params['m1'], m1, s0, a_seq)
        preds_m2 = rollout_mk(models_params['m2'], m2, s0, a_seq, k=4)
        preds_m3 = rollout_mk(models_params['m3'], m3, s0, a_seq, k=8)
        
        # Calculate normalized MSE according to document: E(h) = ||\hat{s}_{t+h} - s_{t+h}||_2 / std
        err_m1 = np.linalg.norm(preds_m1 - target_seq, axis=-1) / (state_std + 1e-8)
        err_m2 = np.linalg.norm(preds_m2 - target_seq, axis=-1) / (state_std + 1e-8)
        err_m3 = np.linalg.norm(preds_m3 - target_seq, axis=-1) / (state_std + 1e-8)
        
        errors_m1.append(err_m1)
        errors_m2.append(err_m2)
        errors_m3.append(err_m3)
        
    E_m1 = np.mean(errors_m1, axis=0)
    E_m2 = np.mean(errors_m2, axis=0)
    E_m3 = np.mean(errors_m3, axis=0)
    
    h_vals = np.arange(1, eval_H + 1)
    
    # Fit curves for M1
    print("--- Curve fitting for M1 ---")
    
    # Integrator
    try:
        popt_int, _ = curve_fit(integrator_law, h_vals, E_m1, p0=[E_m1[0], actual_L], bounds=([0, 0], [np.inf, np.inf]))
        mse_int = np.mean((E_m1 - integrator_law(h_vals, *popt_int))**2)
        aic_int = calculate_aic(eval_H, mse_int, 2)
        print(f"Integrator Law: eps={popt_int[0]:.4f}, L_fit={popt_int[1]:.4f} -> AIC: {aic_int:.2f}")
    except:
        aic_int = np.inf

    # Linear
    try:
        popt_lin, _ = curve_fit(linear_law, h_vals, E_m1, p0=[E_m1[0]])
        mse_lin = np.mean((E_m1 - linear_law(h_vals, *popt_lin))**2)
        aic_lin = calculate_aic(eval_H, mse_lin, 1)
        print(f"Linear Law: c={popt_lin[0]:.4f} -> AIC: {aic_lin:.2f}")
    except:
        aic_lin = np.inf

    # Power
    try:
        popt_pow, _ = curve_fit(power_law, h_vals, E_m1, p0=[E_m1[0], 1.0])
        mse_pow = np.mean((E_m1 - power_law(h_vals, *popt_pow))**2)
        aic_pow = calculate_aic(eval_H, mse_pow, 2)
        print(f"Power Law: c={popt_pow[0]:.4f}, p={popt_pow[1]:.4f} -> AIC: {aic_pow:.2f}")
    except:
        aic_pow = np.inf
        
    if aic_int < aic_lin and aic_int < aic_pow:
        print("=> Integrator Law won the AIC test!")
        diff_L = abs(popt_int[1] - actual_L) / actual_L
        print(f"=> L_fit differs from measured L by {diff_L*100:.2f}%")
        if diff_L <= 0.2:
            print("=> Falsifier passed: L_fit is within 20% of measured L.")
        else:
            print("=> Falsifier triggered: L_fit is NOT within 20% of measured L.")
            
    # Plotting
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(h_vals, E_m1, marker='o', label='M1 (1-step)')
    ax.plot(h_vals, E_m2, marker='s', label='M2 (4-step)')
    ax.plot(h_vals, E_m3, marker='^', label='M3 (8-step)')
    
    if aic_int != np.inf:
        ax.plot(h_vals, integrator_law(h_vals, *popt_int), 'r--', label=f'Integrator Fit (L={popt_int[1]:.2f})')
        
    ax.set_yscale('log')
    ax.set_xlabel('Horizon (h)')
    ax.set_ylabel('E(h) [Normalized Error]')
    ax.set_title(f'Rollout Drift for Linear System (L={actual_L:.2f})')
    ax.legend()
    
    save_plot(fig, f"e1_error_laws_L{args.l_value}.png")

if __name__ == "__main__":
    main()
