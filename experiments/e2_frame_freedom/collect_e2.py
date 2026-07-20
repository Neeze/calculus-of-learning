"""E2 data collection (spec §3.3).

Datasets are collected ONCE and cached, then shared across all regularizer
configs: with a fixed (env, seed) the only difference between runs is the
objective, not the data.

  - train:  one file per (env, seed), used by all 5 configs of that seed.
  - eval:   one ID file + one file per OOD grid condition, collected with a
            dedicated seed (default 99), shared by every run. Eval files also
            store ground-truth (qpos, qvel) for the linear probe.

All files are trajectory-structured (n_traj, T, ...) so that the 10-step
open-loop rollout metric and trajectory-based train/val splits are possible.
"""
import os
import argparse
import numpy as np

from src.envs.dmc_ood import make_dmc_env, OOD_GRID

DATA_DIR = "outputs/data/e2"
CHUNK_LEN = 250  # steps per stored trajectory chunk


def collect_trajectories(env, num_steps: int, rng: np.random.Generator, chunk_len: int = CHUNK_LEN):
    """Random uniform policy, seeded. Episodes are split into fixed-length
    chunks; obs has one more entry than actions (s_0..s_T vs a_0..a_{T-1})."""
    spec = env.action_spec()
    lo, hi = spec.minimum, spec.maximum

    chunks = {"obs": [], "actions": [], "rewards": [], "phys": []}
    n_chunks = num_steps // chunk_len

    s = env.reset()
    p = env.get_physics_state()
    for _ in range(n_chunks):
        obs_buf, act_buf, rew_buf, phys_buf = [s], [], [], [p]
        for _ in range(chunk_len):
            a = rng.uniform(lo, hi, size=spec.shape)
            s, r, done, _ = env.step(a)
            p = env.get_physics_state()
            act_buf.append(a)
            rew_buf.append(r)
            obs_buf.append(s)
            phys_buf.append(p)
            if done:
                s = env.reset()
                p = env.get_physics_state()
                break
        if len(act_buf) < chunk_len:
            continue  # drop short chunk at episode boundary
        chunks["obs"].append(np.array(obs_buf))
        chunks["actions"].append(np.array(act_buf))
        chunks["rewards"].append(np.array(rew_buf))
        chunks["phys"].append(np.array(phys_buf))

    return {k: np.stack(v).astype(np.float32) for k, v in chunks.items()}


def collect_random_dataset(env, num_samples: int = 10000, seed: int = 0):
    """Flat transition dataset with a seeded random policy (used by E3)."""
    data = collect_trajectories(env, num_samples, np.random.default_rng(seed))
    return flatten_transitions(data)


def dataset_path(env_name: str, split: str, seed: int = None, condition: str = None):
    if split == "train":
        return os.path.join(DATA_DIR, f"{env_name}_train_seed{seed}.npz")
    return os.path.join(DATA_DIR, f"{env_name}_eval_{condition}.npz")


def load_dataset(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def flatten_transitions(data):
    """(n_traj, T+1/T, ...) -> flat transition arrays for one-step training/eval."""
    obs, actions, rewards = data["obs"], data["actions"], data["rewards"]
    return {
        "states": obs[:, :-1].reshape(-1, obs.shape[-1]),
        "actions": actions.reshape(-1, actions.shape[-1]),
        "next_states": obs[:, 1:].reshape(-1, obs.shape[-1]),
        "rewards": rewards.reshape(-1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="walker-walk")
    parser.add_argument("--train_seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval_seed", type=int, default=99)
    parser.add_argument("--train_steps", type=int, default=20000)
    parser.add_argument("--eval_steps", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    domain, task = args.env.split("-")

    for seed in args.train_seeds:
        path = dataset_path(args.env, "train", seed=seed)
        if os.path.exists(path):
            print(f"[skip] {path}")
            continue
        print(f"Collecting train data: {args.env} seed={seed}")
        env = make_dmc_env(domain, task, mode="train", seed=seed)
        data = collect_trajectories(env, args.train_steps, np.random.default_rng(seed))
        np.savez_compressed(path, **data)
        print(f"[saved] {path} obs={data['obs'].shape}")

    conditions = {"id": "train", **{k: k for k in OOD_GRID}}
    for cond, mode in conditions.items():
        path = dataset_path(args.env, "eval", condition=cond)
        if os.path.exists(path):
            print(f"[skip] {path}")
            continue
        print(f"Collecting eval data: {args.env} condition={cond}")
        env = make_dmc_env(domain, task, mode=mode, seed=args.eval_seed)
        data = collect_trajectories(env, args.eval_steps, np.random.default_rng(args.eval_seed))
        np.savez_compressed(path, **data)
        print(f"[saved] {path} obs={data['obs'].shape}")


if __name__ == "__main__":
    main()
