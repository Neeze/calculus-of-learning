"""E2 shift-strength oracle (spec §6.2-6.3). Run BEFORE the sweep.

For each OOD condition: take (state, action) pairs sampled under default
physics, reset both a default and a shifted env to the SAME simulator state,
apply the SAME action, and measure the one-step next-observation gap. This is
the true dynamics gap the world models are asked to generalize across.

Interpretation (printed): gap ~ 0 for every condition -> shift too weak, no
signal to discriminate configs (increase shifts); gap enormous everywhere ->
resolution lost at the strong end (rely on the ±15% levels).
"""
import argparse
import numpy as np

from src.envs.dmc_ood import make_dmc_env, OOD_GRID


def dynamics_gap(env_ref, env_shift, n_probes: int, rng: np.random.Generator,
                 warmup_low: int = 5, warmup_high: int = 60):
    spec = env_ref.action_spec()
    lo, hi = spec.minimum, spec.maximum

    gaps = []
    obs_norm = []
    for _ in range(n_probes):
        # Reach a random on-distribution state under default physics.
        env_ref.reset()
        for _ in range(int(rng.integers(warmup_low, warmup_high))):
            env_ref.step(rng.uniform(lo, hi, size=spec.shape))
        qpos = np.array(env_ref.env.physics.data.qpos)
        qvel = np.array(env_ref.env.physics.data.qvel)
        a = rng.uniform(lo, hi, size=spec.shape)

        env_ref.set_physics_state(qpos, qvel)
        s_ref, _, _, _ = env_ref.step(a)

        env_shift.reset()
        env_shift.set_physics_state(qpos, qvel)
        s_shift, _, _, _ = env_shift.step(a)

        gaps.append(np.linalg.norm(s_ref - s_shift))
        obs_norm.append(np.linalg.norm(s_ref))

    return float(np.mean(gaps)), float(np.mean(gaps) / (np.mean(obs_norm) + 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="walker-walk")
    parser.add_argument("--n_probes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    domain, task = args.env.split("-")
    rng = np.random.default_rng(args.seed)
    env_ref = make_dmc_env(domain, task, mode="train", seed=args.seed)

    print(f"=== Shift-strength oracle: {args.env}, {args.n_probes} probes/condition ===")
    print(f"{'condition':10s} {'abs_gap':>10s} {'rel_gap':>10s}")
    rel_gaps = {}
    for cond in OOD_GRID:
        env_shift = make_dmc_env(domain, task, mode=cond, seed=args.seed)
        abs_gap, rel_gap = dynamics_gap(env_ref, env_shift, args.n_probes, rng)
        rel_gaps[cond] = rel_gap
        print(f"{cond:10s} {abs_gap:10.4f} {rel_gap:10.4%}")

    max_rel = max(rel_gaps.values())
    if max_rel < 0.005:
        print("\n[WARN] All conditions produce <0.5% relative one-step gap: the "
              "shift grid is too weak to discriminate configs (spec §6.2). "
              "Increase shift magnitudes before running the sweep.")
    elif min(rel_gaps.values()) > 0.5:
        print("\n[WARN] Even the mildest condition produces >50% relative gap: "
              "resolution may be lost (spec §6.3). Consider adding milder levels.")
    else:
        print("\n[OK] Shift grid spans a measurable, non-degenerate dynamics gap.")


if __name__ == "__main__":
    main()
