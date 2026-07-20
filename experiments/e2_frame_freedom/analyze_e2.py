"""E2 analysis (spec §4c, §10): statistics and figures from results.json.

  P2a  manipulation check — does the regularizer axis move measured frame
       freedom? (Spearman: config rank vs delta/identity_dev over pairs)
  P2b  main test — does measured freedom predict OOD generalization?
       (per-run scatter delta vs OOD ratio, Spearman + bootstrap CI)
  P2c  endpoint ordering — full_rec vs sigreg, paired by seed, Holm-corrected.

Collapse handling: runs with effective rank < latent_dim/8 are flagged; P2b is
reported both with and without them (spec §4a, §6.1).
"""
import os
import json
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.visualization.metrics import (
    spearman_correlation, paired_bootstrap_test, holm_correction,
)
from src.visualization.plotter import save_plot

RESULTS_DIR = "outputs/results/e2"
CONFIG_ORDER = ["full_rec", "light_rec", "vicreg", "light_vicreg", "sigreg"]
PRIMARY_FREEDOM = "delta_ortho"      # pre-registered primary (spec §4c.2)
PRIMARY_OOD = "err_10step"           # pre-registered primary (spec §4c.2)
FREEDOM_KEYS = ["r2_linear", "r2_ortho", "delta_ortho", "identity_dev", "cka"]
OOD_RATIO_KEYS = ["err_1step", "err_10step", "err_rollout_auc", "reward_mse"]


def iqm(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    lo, hi = int(np.floor(n * 0.25)), int(np.ceil(n * 0.75))
    return float(np.mean(x[lo:hi])) if hi > lo else float(np.mean(x))


def bootstrap_ci(x, stat=np.mean, n_boot=10000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    boots = [stat(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def bootstrap_spearman_ci(x, y, n_boot=5000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        rx = np.argsort(np.argsort(x[idx])).astype(float)
        ry = np.argsort(np.argsort(y[idx])).astype(float)
        rhos.append(np.corrcoef(rx, ry)[0, 1])
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[table] {path}")


def run_freedom(run, pairs):
    """Per-run freedom = mean over pairs of the same config involving its seed."""
    vals = [p[PRIMARY_FREEDOM] for p in pairs
            if p["reg"] == run["reg"] and run["seed"] in (p["seed_i"], p["seed_j"])]
    return float(np.mean(vals)) if vals else np.nan


def analyze(results, out_dir):
    env = results["env"]
    d = results["latent_dim"]
    runs, pairs = results["runs"], results["pairs"]
    collapse_thresh = d / 8.0

    for r in runs:
        r["collapsed"] = r["eff_rank"] < collapse_thresh
        r["freedom"] = run_freedom(r, pairs)

    n_collapsed = sum(r["collapsed"] for r in runs)
    print(f"=== {env}: {len(runs)} runs, {len(pairs)} pairs, "
          f"{n_collapsed} collapsed (eff_rank < {collapse_thresh:.0f}) ===")

    # ---------------- P2a: manipulation check ----------------
    print("\n--- P2a: regularizer axis vs measured frame freedom ---")
    rows = []
    for reg in CONFIG_ORDER:
        cfg_pairs = [p for p in pairs if p["reg"] == reg]
        if not cfg_pairs:
            continue
        med = {k: float(np.median([p[k] for p in cfg_pairs])) for k in FREEDOM_KEYS}
        rows.append([reg] + [f"{med[k]:.4f}" for k in FREEDOM_KEYS])
        print(f"{reg:14s} " + "  ".join(f"{k}={med[k]:.3f}" for k in FREEDOM_KEYS))
    write_csv(os.path.join(out_dir, f"table_p2a_{env}.csv"),
              ["config"] + FREEDOM_KEYS, rows)

    axis_rank = [CONFIG_ORDER.index(p["reg"]) for p in pairs]
    p2a_stats = {}
    for key in ("delta_ortho", "identity_dev"):
        rho, pval = spearman_correlation(axis_rank, [p[key] for p in pairs])
        p2a_stats[key] = {"rho": rho, "p": pval}
        print(f"Spearman(config rank, {key}): rho={rho:.3f}, p={pval:.4f}  (n={len(pairs)})")

    # ---------------- P2b: freedom vs OOD (main test) ----------------
    print("\n--- P2b: measured freedom vs OOD ratio (primary: "
          f"{PRIMARY_FREEDOM} vs {PRIMARY_OOD}) ---")
    p2b_stats = {}
    for subset_name, subset in (("all", runs),
                                ("no_collapsed", [r for r in runs if not r["collapsed"]])):
        if len(subset) < 4:
            print(f"[{subset_name}] too few runs ({len(subset)}), skipping")
            continue
        x = [r["freedom"] for r in subset]
        y = [r["ood_ratio_mean"][PRIMARY_OOD] for r in subset]
        rho, pval = spearman_correlation(x, y)
        ci = bootstrap_spearman_ci(x, y)
        slope = float(np.polyfit(x, y, 1)[0])
        p2b_stats[subset_name] = {"rho": rho, "p": pval, "ci": ci,
                                  "ols_slope": slope, "n": len(subset)}
        print(f"[{subset_name}] n={len(subset)}: rho={rho:.3f} (95% CI [{ci[0]:.3f}, "
              f"{ci[1]:.3f}]), p={pval:.4f}, OLS slope={slope:.3f}")

    # Secondary OOD metrics on all runs
    for key in OOD_RATIO_KEYS + ["probe"]:
        y = ([r["ood_probe_r2_mean"] for r in runs] if key == "probe"
             else [r["ood_ratio_mean"][key] for r in runs])
        rho, pval = spearman_correlation([r["freedom"] for r in runs], y)
        print(f"  secondary [{key}]: rho={rho:.3f}, p={pval:.4f}")

    # ---------------- P2c: endpoint ordering ----------------
    print("\n--- P2c: full_rec vs sigreg (paired by seed) ---")
    p2c_stats, pvals, keys = {}, [], []
    by = lambda reg: {r["seed"]: r for r in runs if r["reg"] == reg}
    fr, sg = by("full_rec"), by("sigreg")
    common = sorted(set(fr) & set(sg))
    if len(common) >= 2:
        for key in OOD_RATIO_KEYS:
            a = [fr[s]["ood_ratio_mean"][key] for s in common]  # expect higher (worse)
            b = [sg[s]["ood_ratio_mean"][key] for s in common]
            diff, p = paired_bootstrap_test(a, b)
            p2c_stats[key] = {"mean_diff_fullrec_minus_sigreg": diff, "p_raw": p}
            pvals.append(p)
            keys.append(key)
        a = [sg[s]["ood_probe_r2_mean"] for s in common]  # probe: higher is better
        b = [fr[s]["ood_probe_r2_mean"] for s in common]
        diff, p = paired_bootstrap_test(a, b)
        p2c_stats["probe_r2"] = {"mean_diff_sigreg_minus_fullrec": diff, "p_raw": p}
        pvals.append(p)
        keys.append("probe_r2")
        adj = holm_correction(pvals)
        for k, p_adj in zip(keys, adj):
            p2c_stats[k]["p_holm"] = float(p_adj)
            print(f"{k:16s} diff={list(p2c_stats[k].values())[0]:+.4f} "
                  f"p_raw={p2c_stats[k]['p_raw']:.4f} p_holm={p_adj:.4f}")
    else:
        print(f"Not enough common seeds ({common}) for paired test.")

    # ---------------- Sanity tables (spec §4b pitfall, §10) ----------------
    rows = [[r["reg"], r["seed"], f"{r['eff_rank']:.1f}", r["collapsed"],
             f"{r['id']['err_1step']:.4f}", f"{r['id']['err_10step']:.4f}",
             f"{r['id']['reward_mse']:.4f}", f"{r['id']['probe_r2']:.3f}",
             f"{r['freedom']:.4f}", f"{r['ood_ratio_mean'][PRIMARY_OOD]:.3f}"]
            for r in runs]
    write_csv(os.path.join(out_dir, f"table_runs_{env}.csv"),
              ["config", "seed", "eff_rank", "collapsed", "id_err1", "id_err10",
               "id_reward_mse", "id_probe_r2", "freedom_delta", "ood_ratio_10step"],
              rows)

    axis_of = lambda c: ("mass" if c.startswith("mass") else
                         "fric" if c.startswith("fric") else "comb")
    rows = []
    for reg in CONFIG_ORDER:
        cfg = [r for r in runs if r["reg"] == reg]
        if not cfg:
            continue
        row = [reg]
        for ax in ("mass", "fric", "comb"):
            vals = [r["ood"][c][PRIMARY_OOD] / (r["id"][PRIMARY_OOD] + 1e-12)
                    for r in cfg for c in r["ood"] if axis_of(c) == ax]
            row.append(f"{iqm(vals):.3f}")
        rows.append(row)
    write_csv(os.path.join(out_dir, f"table_ood_breakdown_{env}.csv"),
              ["config", "mass_iqm", "fric_iqm", "comb_iqm"], rows)

    # ---------------- Figures ----------------
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    xs = np.arange(len(CONFIG_ORDER))
    for key, marker in (("delta_ortho", "o"), ("identity_dev", "s")):
        med = [np.median([p[key] for p in pairs if p["reg"] == c] or [np.nan])
               for c in CONFIG_ORDER]
        axes[0].plot(xs, med, marker=marker, markersize=4, label=key)
    for key, marker in (("r2_linear", "o"), ("r2_ortho", "s"), ("cka", "^")):
        med = [np.median([p[key] for p in pairs if p["reg"] == c] or [np.nan])
               for c in CONFIG_ORDER]
        axes[1].plot(xs, med, marker=marker, markersize=4, label=key)
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(CONFIG_ORDER, rotation=30, ha="right")
        ax.legend(fontsize=6)
    axes[0].set_ylabel("deviation")
    axes[1].set_ylabel("similarity")
    fig.suptitle(f"E2-A ({env}): frame freedom along regularizer axis")
    save_plot(fig, f"e2A_freedom_axis_{env}.png")

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    colors = dict(zip(CONFIG_ORDER, ["#e34a33", "#fc8d59", "#2c7fb8", "#7fcdbb", "#31a354"]))
    for r in runs:
        c = colors[r["reg"]]
        ax.scatter(r["freedom"], r["ood_ratio_mean"][PRIMARY_OOD], s=22,
                   facecolors="none" if r["collapsed"] else c, edgecolors=c,
                   label=r["reg"])
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=6)
    if "all" in p2b_stats:
        s = p2b_stats["all"]
        ax.set_title(f"rho={s['rho']:.2f} [{s['ci'][0]:.2f},{s['ci'][1]:.2f}], p={s['p']:.3f}",
                     fontsize=7)
    ax.set_xlabel(f"frame freedom ({PRIMARY_FREEDOM})")
    ax.set_ylabel(f"OOD/ID ratio ({PRIMARY_OOD})")
    save_plot(fig, f"e2B_freedom_vs_ood_{env}.png")

    stats = {"env": env, "collapse_threshold": collapse_thresh,
             "n_collapsed": n_collapsed, "p2a": p2a_stats, "p2b": p2b_stats,
             "p2c": p2c_stats}
    with open(os.path.join(out_dir, f"stats_{env}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved stats_{env}.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="walker-walk")
    args = parser.parse_args()

    path = os.path.join(RESULTS_DIR, f"results_{args.env}.json")
    with open(path) as f:
        results = json.load(f)
    analyze(results, RESULTS_DIR)


if __name__ == "__main__":
    main()
