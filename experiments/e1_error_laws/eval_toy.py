import os
import argparse
import pickle
import numpy as np
import jax.numpy as jnp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from src.models.mlp_predictors import create_m1_model, create_mk_model
from src.visualization.metrics import calculate_aic, paired_bootstrap_test, holm_correction
from src.visualization.plotter import save_plot

L_REGIME_TOL = 0.02          # |L-1| below this is treated as the critical regime
PRE_OVERFLOW_H = 25          # per spec §5.3.4: 1.2^25 ~ 95x, exponential visible, no overflow

def rollout_m1(params, model, start_state, actions):
    """One-step (Euler-analogue) predictor: composes h times for horizon h."""
    H = actions.shape[0]
    preds = []
    curr_state = start_state
    for t in range(H):
        curr_state = np.array(model.apply({'params': params}, curr_state, actions[t]))
        preds.append(curr_state)
    return np.array(preds)

def rollout_block(params, model, start_state, actions, k):
    """
    Block (higher-order) predictor: one forward pass predicts a whole block
    of k future states from k actions. Rollout jumps by up to k steps at a
    time and only composes ceil(H/k) times, not H times (E1 spec §3.3/§3.4).
    """
    H = actions.shape[0]
    d_action = actions.shape[-1]
    preds = []
    curr_state = start_state
    t = 0
    while t < H:
        n = min(k, H - t)
        block_actions = actions[t:t + k]
        if block_actions.shape[0] < k:
            pad = np.zeros((k - block_actions.shape[0], d_action))
            block_actions = np.concatenate([block_actions, pad], axis=0)
        pred_block = np.array(model.apply({'params': params}, curr_state, block_actions))
        preds.extend(pred_block[:n])
        curr_state = pred_block[n - 1]
        t += n
    return np.array(preds)

def rollout_oracle(A, B, start_state, actions):
    """Ground-truth dynamics rollout (no model). This is the noise floor."""
    preds, s = [], start_state
    for a in actions:
        s = A @ s + B @ a
        preds.append(s)
    return np.array(preds)

def integrator_law(h, eps, L):
    L_safe = np.where(np.abs(L - 1.0) < 1e-6, 1.0 + 1e-6, L)
    return eps * (np.power(L_safe, h) - 1.0) / (L_safe - 1.0)

def linear_law(h, c):
    return c * h

def power_law(h, c, p):
    return c * (h ** p)

def fit_and_report(E, h_vals, actual_L, label):
    """
    Fit E(h) to the integrator/linear/power-law families and check the
    regime-appropriate criterion (E1 spec §2, §4.4):
      - L<1 (contractive): integrator law must win AIC AND the measured
        plateau must match eps/(1-L_true) within 20%.
      - L>=1 (critical/expansive): integrator law must win AIC AND the
        growth rate (slope of log E(h)) must match log(L_true), fit on the
        pre-overflow horizon only.
    Fitting never silently swallows exceptions into AIC=inf (bug #7): any
    curve_fit failure is logged with the underlying error.
    """
    is_expansive = actual_L >= 1.0 - L_REGIME_TOL
    if is_expansive:
        fit_mask = h_vals <= PRE_OVERFLOW_H
    else:
        fit_mask = np.ones_like(h_vals, dtype=bool)
    h_fit = h_vals[fit_mask]
    E_fit = E[fit_mask]

    results = {'label': label, 'regime': 'expansive_or_critical' if is_expansive else 'contractive'}

    try:
        popt_int, _ = curve_fit(
            integrator_law, h_fit, E_fit, p0=[E_fit[0], actual_L],
            bounds=([0, 0], [np.inf, np.inf]), maxfev=10000,
        )
        mse_int = np.mean((E_fit - integrator_law(h_fit, *popt_int)) ** 2)
        aic_int = calculate_aic(len(h_fit), mse_int, 2)
        print(f"[{label}] Integrator Law: eps={popt_int[0]:.4f}, L_fit={popt_int[1]:.4f} -> AIC: {aic_int:.2f}")
    except Exception as e:
        print(f"[{label}] Integrator Law fit FAILED: {e}")
        popt_int, aic_int = None, np.inf

    try:
        popt_lin, _ = curve_fit(linear_law, h_fit, E_fit, p0=[E_fit[0]], maxfev=10000)
        mse_lin = np.mean((E_fit - linear_law(h_fit, *popt_lin)) ** 2)
        aic_lin = calculate_aic(len(h_fit), mse_lin, 1)
        print(f"[{label}] Linear Law: c={popt_lin[0]:.4f} -> AIC: {aic_lin:.2f}")
    except Exception as e:
        print(f"[{label}] Linear Law fit FAILED: {e}")
        popt_lin, aic_lin = None, np.inf

    try:
        popt_pow, _ = curve_fit(power_law, h_fit, E_fit, p0=[E_fit[0], 1.0], maxfev=10000)
        mse_pow = np.mean((E_fit - power_law(h_fit, *popt_pow)) ** 2)
        aic_pow = calculate_aic(len(h_fit), mse_pow, 2)
        print(f"[{label}] Power Law: c={popt_pow[0]:.4f}, p={popt_pow[1]:.4f} -> AIC: {aic_pow:.2f}")
    except Exception as e:
        print(f"[{label}] Power Law fit FAILED: {e}")
        popt_pow, aic_pow = None, np.inf

    integrator_wins = aic_int < aic_lin and aic_int < aic_pow
    results.update({
        'popt_int': popt_int, 'aic_int': aic_int,
        'popt_lin': popt_lin, 'aic_lin': aic_lin,
        'popt_pow': popt_pow, 'aic_pow': aic_pow,
        'integrator_wins': integrator_wins,
    })

    if is_expansive:
        # Growth-rate check: slope of log E(h) vs h on the pre-overflow segment.
        valid = E_fit > 0
        if valid.sum() >= 2:
            slope, _ = np.polyfit(h_fit[valid], np.log(E_fit[valid]), 1)
            theory = np.log(actual_L)
            rel_err = abs(slope - theory) / abs(theory) if theory != 0 else np.inf
            print(f"[{label}] Growth rate (slope of log E): {slope:.4f} vs log(L_true)={theory:.4f} "
                  f"-> rel. error {rel_err*100:.2f}%")
            results.update({'growth_rate_fit': slope, 'growth_rate_theory': theory, 'regime_rel_err': rel_err})
        else:
            print(f"[{label}] Growth rate check skipped: not enough positive E(h) points.")
            results.update({'growth_rate_fit': None, 'growth_rate_theory': None, 'regime_rel_err': np.inf})
    else:
        eps_ref = popt_int[0] if popt_int is not None else E[0]
        plateau_theory = eps_ref / (1.0 - actual_L)
        plateau_measured = np.mean(E[-max(1, len(E) // 5):])
        rel_err = abs(plateau_measured - plateau_theory) / plateau_theory if plateau_theory != 0 else np.inf
        print(f"[{label}] Plateau measured: {plateau_measured:.4f} vs theory eps/(1-L)={plateau_theory:.4f} "
              f"-> rel. error {rel_err*100:.2f}%")
        results.update({'plateau_measured': plateau_measured, 'plateau_theory': plateau_theory, 'regime_rel_err': rel_err})

    results['law_confirmed'] = bool(integrator_wins and results['regime_rel_err'] < 0.20)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--l_values', type=float, nargs='+', default=[0.8, 0.95, 1.05, 1.2])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    args = parser.parse_args()

    d = 8
    m1 = create_m1_model(d=d)
    m2 = create_mk_model(d=d, k=4)
    m3 = create_mk_model(d=d, k=8)
    model_specs = [('M1', m1, 1), ('M2', m2, 4), ('M3', m3, 8)]

    hstar_rows = []
    aic_rows = []
    regime_rows = []

    for L in args.l_values:
        print(f"\n========== Evaluating L={L} ==========")
        all_E = {name: [] for name, _, _ in model_specs}
        all_E_oracle = []
        hstar_by_model = {name: [] for name, _, _ in model_specs}
        actual_L = None

        for seed in args.seeds:
            with open(f"outputs/checkpoints/e1/test_data_L{L}_seed{seed}.pkl", "rb") as f:
                data = pickle.load(f)
                test_data = data['test_data']
                A_matrix = data['A_matrix']
                B_matrix = data['B_matrix']
                state_std = data['state_std']

            actual_L = np.linalg.norm(A_matrix, ord=2)

            with open(f"outputs/checkpoints/e1/models_L{L}_seed{seed}.pkl", "rb") as f:
                models_params = pickle.load(f)

            states = test_data['states']
            actions = test_data['actions']
            num_trajs, H, _ = states.shape

            eval_H = min(50, H - 1)
            h_vals = np.arange(1, eval_H + 1)

            errors = {name: [] for name, _, _ in model_specs}
            errors_oracle = []

            for i in range(num_trajs):
                s0 = states[i, 0]
                a_seq = actions[i, :eval_H]
                target_seq = test_data['next_states'][i, :eval_H]

                for name, model, k in model_specs:
                    params = models_params[name.lower()]
                    if k == 1:
                        preds = rollout_m1(params, model, s0, a_seq)
                    else:
                        preds = rollout_block(params, model, s0, a_seq, k)
                    err = np.linalg.norm(preds - target_seq, axis=-1) / (state_std + 1e-8)
                    errors[name].append(err)

                preds_oracle = rollout_oracle(A_matrix, B_matrix, s0, a_seq)
                err_oracle = np.linalg.norm(preds_oracle - target_seq, axis=-1) / (state_std + 1e-8)
                errors_oracle.append(err_oracle)

            E_oracle_seed = np.mean(errors_oracle, axis=0)
            all_E_oracle.append(E_oracle_seed)

            # h*(tau): tau relative to the noise floor (oracle). For L<1 the
            # oracle plateaus, so tau = 2 * plateau_oracle. For L>=1 there is
            # no plateau (noise also compounds), so tau = 2 * one-step
            # oracle error is used as the reference noise-floor scale.
            if actual_L < 1.0 - L_REGIME_TOL:
                tau_seed = 2.0 * np.mean(E_oracle_seed[-max(1, eval_H // 5):])
            else:
                tau_seed = 2.0 * E_oracle_seed[0]

            for name, _, _ in model_specs:
                E_model_seed = np.mean(errors[name], axis=0)
                all_E[name].append(E_model_seed)
                above = np.where(E_model_seed > tau_seed)[0]
                hstar = int(h_vals[above[0]]) if len(above) > 0 else int(eval_H)  # right-censored at eval_H
                hstar_by_model[name].append(hstar)

        E_oracle = np.mean(all_E_oracle, axis=0)
        E = {name: np.mean(all_E[name], axis=0) for name, _, _ in model_specs}
        E_std = {name: np.std(all_E[name], axis=0) for name, _, _ in model_specs}
        E_excess = {name: np.clip(E[name] - E_oracle, a_min=0.0, a_max=None) for name, _, _ in model_specs}

        h_vals = np.arange(1, eval_H + 1)

        print(f"--- Curve fitting on excess error (E_model - E_oracle) ---")
        fit_results = {}
        for name, _, _ in model_specs:
            fit_results[name] = fit_and_report(E_excess[name], h_vals, actual_L, label=name)
            aic_rows.append((L, name, fit_results[name]['aic_int'], fit_results[name]['aic_lin'],
                              fit_results[name]['aic_pow'], fit_results[name]['integrator_wins']))
            regime_rows.append((L, name, fit_results[name]['regime'], fit_results[name]['regime_rel_err'],
                                 fit_results[name]['law_confirmed']))

        # h*(tau): paired bootstrap M2/M3 vs M1 across seeds.
        p_raw = []
        comparisons = []
        for name in ['M2', 'M3']:
            mean_diff, p = paired_bootstrap_test(hstar_by_model[name], hstar_by_model['M1'])
            p_raw.append(p)
            comparisons.append(name)
        p_adj = holm_correction(p_raw)
        for name, p_adj_val in zip(comparisons, p_adj):
            mean_h1 = np.mean(hstar_by_model['M1'])
            mean_hk = np.mean(hstar_by_model[name])
            print(f"[h*] {name} vs M1: mean h*={mean_hk:.2f} vs {mean_h1:.2f}, Holm-adj p={p_adj_val:.4f}")
            hstar_rows.append((L, name, mean_h1, mean_hk, p_adj_val, p_adj_val < 0.05))

        fig, ax = plt.subplots(figsize=(8, 6))
        markers = {'M1': 'o', 'M2': 's', 'M3': '^'}
        for name, _, _ in model_specs:
            ax.plot(h_vals, E[name], marker=markers[name], label=f'{name}')
            ax.fill_between(h_vals, E[name] - E_std[name], E[name] + E_std[name], alpha=0.2)
        ax.plot(h_vals, E_oracle, 'k:', label='Oracle (noise floor)')

        popt_m1 = fit_results['M1']['popt_int']
        if popt_m1 is not None:
            ax.plot(h_vals, integrator_law(h_vals, *popt_m1) + E_oracle, 'r--',
                     label=f"M1 Integrator Fit (L={popt_m1[1]:.2f})")

        ax.set_yscale('log')
        ax.set_xlabel('Horizon (h)')
        ax.set_ylabel('E(h) [Normalized Error]')
        ax.set_title(f'Rollout Drift for Linear System (L={actual_L:.2f})')
        ax.legend()

        os.makedirs("outputs/plots", exist_ok=True)
        save_plot(fig, f"e1_error_laws_L{L}.png")

    print("\n========== AIC table ==========")
    print(f"{'L':>6} {'model':>6} {'AIC_int':>10} {'AIC_lin':>10} {'AIC_pow':>10} {'int_wins':>9}")
    for row in aic_rows:
        print(f"{row[0]:6.2f} {row[1]:>6} {row[2]:10.2f} {row[3]:10.2f} {row[4]:10.2f} {str(row[5]):>9}")

    print("\n========== Regime-fit table ==========")
    print(f"{'L':>6} {'model':>6} {'regime':>22} {'rel_err':>10} {'law_confirmed':>14}")
    for row in regime_rows:
        print(f"{row[0]:6.2f} {row[1]:>6} {row[2]:>22} {row[3]*100:9.2f}% {str(row[4]):>14}")

    print("\n========== h*(tau) table ==========")
    print(f"{'L':>6} {'model':>6} {'mean_h*_M1':>11} {'mean_h*':>9} {'p_holm':>8} {'p<0.05':>7}")
    for row in hstar_rows:
        print(f"{row[0]:6.2f} {row[1]:>6} {row[2]:11.2f} {row[3]:9.2f} {row[4]:8.4f} {str(row[5]):>7}")

    with open("outputs/plots/e1_aic_table.csv", "w") as f:
        f.write("L,model,aic_int,aic_lin,aic_pow,integrator_wins\n")
        for row in aic_rows:
            f.write(",".join(str(x) for x in row) + "\n")
    with open("outputs/plots/e1_regime_fit_table.csv", "w") as f:
        f.write("L,model,regime,rel_err,law_confirmed\n")
        for row in regime_rows:
            f.write(",".join(str(x) for x in row) + "\n")
    with open("outputs/plots/e1_hstar_table.csv", "w") as f:
        f.write("L,model,mean_hstar_m1,mean_hstar,p_holm,significant\n")
        for row in hstar_rows:
            f.write(",".join(str(x) for x in row) + "\n")

if __name__ == "__main__":
    main()
