"""E2 evaluation (spec §4a-4b, §7): raw measurements -> results.json.

Per run (config x seed):
  - 1-step and 10-step open-loop latent prediction error, normalized by the
    run's own ID latent variance (scale-free, comparable as OOD/ID ratio only)
  - reward prediction MSE (shared space, absolute values comparable)
  - ridge probe z -> (qpos, qvel), fit on ID, frozen, evaluated on ID holdout
    and on every OOD condition
  - effective rank of ID latents (collapse diagnostic)

Per seed-pair within a config:
  - frame-freedom metrics (R^2 linear/ortho, delta, identity dev, CKA) on one
    FIXED shared observation set (spec §6.5)

No statistics here — analyze_e2.py consumes results.json.
"""
import os
import json
import argparse
import pickle
import itertools
import numpy as np
import jax
import jax.numpy as jnp

from src.envs.dmc_ood import OOD_GRID
from experiments.e2_frame_freedom.train_e2 import WorldModel, REGULARIZERS
from experiments.e2_frame_freedom.collect_e2 import dataset_path, load_dataset, flatten_transitions
from src.visualization.metrics import (
    frame_freedom_metrics, effective_rank, ridge_probe_r2,
)

CKPT_DIR = "outputs/checkpoints/e2"
RESULTS_DIR = "outputs/results/e2"
ROLLOUT_H = 10
FREEDOM_N = 5000  # fixed number of shared observations for frame-freedom metrics


def load_run(env_name, reg, seed):
    tag = f"{env_name}_{reg}_seed{seed}"
    with open(os.path.join(CKPT_DIR, f"model_{tag}.pkl"), "rb") as f:
        params = pickle.load(f)
    with open(os.path.join(CKPT_DIR, f"config_{tag}.json")) as f:
        config = json.load(f)
    return params, config


def make_model_fns(model, params):
    @jax.jit
    def encode(x):
        return model.apply({"params": params}, x, method=WorldModel.encode)

    @jax.jit
    def predict(z, a):
        return model.apply({"params": params}, z, a, method=WorldModel.predict)

    @jax.jit
    def predict_reward(z, a):
        return model.apply({"params": params}, z, a, method=WorldModel.predict_reward)

    return encode, predict, predict_reward


def rollout_windows(data, horizon, stride=10):
    """Slice trajectory chunks into (start-state, action window, target obs
    window) triples for open-loop rollout evaluation."""
    obs, actions = data["obs"], data["actions"]
    E, T = actions.shape[0], actions.shape[1]
    starts = np.arange(0, T - horizon + 1, stride)
    obs0, act_w, obs_w = [], [], []
    for t in starts:
        obs0.append(obs[:, t])
        act_w.append(actions[:, t:t + horizon])
        obs_w.append(obs[:, t + 1:t + horizon + 1])
    return (np.concatenate(obs0, axis=0),
            np.concatenate(act_w, axis=0),
            np.concatenate(obs_w, axis=0))


def evaluate_run_on_dataset(fns, data, latent_var):
    """All per-dataset metrics for one run. latent_var is the run's ID latent
    variance (computed once, on ID); passing it in keeps OOD normalization
    consistent with ID normalization."""
    encode, predict, predict_reward = fns
    flat = flatten_transitions(data)
    s = jnp.array(flat["states"])
    a = jnp.array(flat["actions"])
    s_next = jnp.array(flat["next_states"])
    r = jnp.array(flat["rewards"])

    z = encode(s)
    z_next = encode(s_next)

    err1 = float(jnp.mean(jnp.sum((predict(z, a) - z_next) ** 2, axis=-1))
                 / (latent_var * z.shape[-1]))
    reward_mse = float(jnp.mean((predict_reward(z, a) - r) ** 2))

    obs0, act_w, obs_w = rollout_windows(data, ROLLOUT_H)
    z_roll = encode(jnp.array(obs0))
    errs_h = []
    for h in range(ROLLOUT_H):
        z_roll = predict(z_roll, jnp.array(act_w[:, h]))
        z_true = encode(jnp.array(obs_w[:, h]))
        e = float(jnp.mean(jnp.sum((z_roll - z_true) ** 2, axis=-1))
                  / (latent_var * z_true.shape[-1]))
        errs_h.append(e)

    return {
        "err_1step": err1,
        "err_10step": errs_h[-1],
        "err_rollout_curve": errs_h,
        "err_rollout_auc": float(np.mean(errs_h)),
        "reward_mse": reward_mse,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="walker-walk")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--latent_dim", type=int, default=32)
    args = parser.parse_args()

    print("Loading cached eval datasets...")
    id_data = load_dataset(dataset_path(args.env, "eval", condition="id"))
    ood_data = {c: load_dataset(dataset_path(args.env, "eval", condition=c))
                for c in OOD_GRID}

    obs_dim = id_data["obs"].shape[-1]
    model = WorldModel(obs_dim=obs_dim, latent_dim=args.latent_dim)

    id_flat = flatten_transitions(id_data)
    # Fixed shared observation set for frame freedom — identical for every pair.
    freedom_obs = jnp.array(id_flat["states"][:FREEDOM_N])

    # Probe targets: physics state aligned with obs (per-chunk, obs has T+1 entries).
    def flat_phys(data):
        p = data["phys"][:, :-1]
        return p.reshape(-1, p.shape[-1])

    id_phys = flat_phys(id_data)
    n_fit = int(0.8 * len(id_flat["states"]))

    results = {"env": args.env, "latent_dim": args.latent_dim,
               "conditions": list(OOD_GRID), "runs": [], "pairs": []}
    latents_cache = {}

    for reg in REGULARIZERS:
        for seed in args.seeds:
            try:
                params, config = load_run(args.env, reg, seed)
            except FileNotFoundError:
                print(f"[skip] missing checkpoint {reg} seed {seed}")
                continue
            print(f"--- Evaluating {reg} seed {seed} ---")
            fns = make_model_fns(model, params)
            encode = fns[0]

            z_id = np.array(encode(jnp.array(id_flat["states"])))
            latents_cache[(reg, seed)] = np.array(encode(freedom_obs))

            # Run's own ID latent scale — the normalizer for ALL of its errors
            # (spec §4b.1: variance, not std).
            latent_var = float(np.mean(np.var(
                np.array(encode(jnp.array(id_flat["next_states"]))), axis=0))) + 1e-12

            id_metrics = evaluate_run_on_dataset(fns, id_data, latent_var)
            id_metrics["probe_r2"] = ridge_probe_r2(
                z_id[:n_fit], id_phys[:n_fit], z_id[n_fit:], id_phys[n_fit:])

            ood_metrics, ratios = {}, {}
            for cond, data in ood_data.items():
                m = evaluate_run_on_dataset(fns, data, latent_var)
                z_ood = np.array(encode(jnp.array(flatten_transitions(data)["states"])))
                m["probe_r2"] = ridge_probe_r2(
                    z_id[:n_fit], id_phys[:n_fit], z_ood, flat_phys(data))
                ood_metrics[cond] = m
                for key in ("err_1step", "err_10step", "err_rollout_auc", "reward_mse"):
                    ratios.setdefault(key, []).append(m[key] / (id_metrics[key] + 1e-12))

            run = {
                "reg": reg,
                "seed": seed,
                "eff_rank": effective_rank(z_id),
                "latent_var": latent_var,
                "best_val_pred_loss": config.get("best_val_pred_loss"),
                "id": id_metrics,
                "ood": ood_metrics,
                "ood_ratio_mean": {k: float(np.mean(v)) for k, v in ratios.items()},
                "ood_probe_r2_mean": float(np.mean(
                    [m["probe_r2"] for m in ood_metrics.values()])),
            }
            results["runs"].append(run)
            print(f"  eff_rank={run['eff_rank']:.1f} | ID err1={id_metrics['err_1step']:.4f} "
                  f"err10={id_metrics['err_10step']:.4f} probeR2={id_metrics['probe_r2']:.3f} | "
                  f"OOD ratio(10-step)={run['ood_ratio_mean']['err_10step']:.3f}")

    for reg in REGULARIZERS:
        seeds_present = [s for s in args.seeds if (reg, s) in latents_cache]
        for si, sj in itertools.combinations(seeds_present, 2):
            m = frame_freedom_metrics(latents_cache[(reg, si)], latents_cache[(reg, sj)])
            results["pairs"].append({"reg": reg, "seed_i": si, "seed_j": sj, **m})
            print(f"pair {reg} ({si},{sj}): R2lin={m['r2_linear']:.3f} "
                  f"R2orth={m['r2_ortho']:.3f} delta={m['delta_ortho']:.3f} "
                  f"iddev={m['identity_dev']:.3f} CKA={m['cka']:.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"results_{args.env}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out} ({len(results['runs'])} runs, {len(results['pairs'])} pairs).")


if __name__ == "__main__":
    main()
