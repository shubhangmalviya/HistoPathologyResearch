import os
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, List
from scipy import stats

from src.utils.paths import results_dir


def rank_biserial_from_diffs(diffs: np.ndarray) -> float:
    diffs = diffs[diffs != 0]
    if diffs.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diffs))
    r_plus = ranks[diffs > 0].sum()
    r_minus = ranks[diffs < 0].sum()
    denom = ranks.sum()
    if denom == 0:
        return 0.0
    return float((r_plus - r_minus) / denom)


def bootstrap_ci(data: np.ndarray, stat_fn, n_boot: int = 2000, alpha: float = 0.05, rng: np.random.Generator | None = None) -> Tuple[float, float]:
    rng = rng or np.random.default_rng(42)
    n = data.shape[0]
    stats_ = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats_.append(stat_fn(data[idx]))
    q_low = 100 * (alpha / 2)
    q_high = 100 * (1 - alpha / 2)
    return float(np.percentile(stats_, q_low)), float(np.percentile(stats_, q_high))


def _bh_adjust(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    p_sorted = np.array(pvals, dtype=float)[order]
    adj = p_sorted * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    result = np.empty_like(adj)
    result[order.argsort()] = adj
    return result.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=os.path.join(results_dir(), "per_image_metrics.csv"))
    parser.add_argument("--out_summary", type=str, default=os.path.join(results_dir(), "summary_rq2.csv"))
    parser.add_argument("--out_bh", type=str, default=os.path.join(results_dir(), "bh_adjusted.csv"))
    parser.add_argument("--out_fig_dir", type=str, default=os.path.join(results_dir(), "figures"))
    parser.add_argument("--out_forest_csv", type=str, default=os.path.join(results_dir(), "forest_pq.csv"))
    parser.add_argument("--out_report_txt", type=str, default=os.path.join(results_dir(), "report_sentences.txt"))
    args = parser.parse_args()

    # Load CSV or Parquet automatically
    inp = args.input
    if inp.lower().endswith('.csv'):
        df = pd.read_csv(inp)
    else:
        df = pd.read_parquet(inp)

    # Pivot to paired wide
    wide = df.pivot_table(index=["image_id", "tissue"], columns="model", values=["PQ", "Dice", "AJI", "F1"]).dropna()

    # Deltas
    for m in ["PQ", "Dice", "AJI", "F1"]:
        wide[(m, "Delta")] = wide[(m, "expert")] - wide[(m, "unified")]

    # Sample sizes
    n_overall = wide.shape[0]
    counts = wide.reset_index().groupby("tissue").size().to_numpy()
    n_tissues = counts.size
    median_nt = int(np.median(counts)) if n_tissues else 0
    range_nt = (int(counts.min()), int(counts.max())) if n_tissues else (0, 0)

    # Primary Wilcoxon (one-sided) on PQ
    d_pq = wide[("PQ", "Delta")].to_numpy()
    wstat, p_two = stats.wilcoxon(d_pq, alternative="two-sided", zero_method="wilcox", mode="auto")
    p_one = p_two / 2 if float(np.mean(d_pq)) > 0 else 1 - p_two / 2
    rb = rank_biserial_from_diffs(d_pq)
    median_delta = float(np.median(d_pq))
    ci_low, ci_high = bootstrap_ci(d_pq, np.median)

    # Mixed effects for tissue clustering
    # Cluster-aware test via cluster means (tissues) one-sample t-test
    df_mixed = wide.copy().reset_index()
    cols = df_mixed.columns
    tissue_key = ("tissue", "") if ("tissue", "") in cols else "tissue"
    pqdelta_key = ("PQ", "Delta") if ("PQ", "Delta") in cols else "PQ_Delta"
    df_mixed = df_mixed[[tissue_key, pqdelta_key]]
    df_mixed.columns = ["tissue", "PQ_Delta"]
    means_by_tissue = df_mixed.groupby("tissue")["PQ_Delta"].mean()
    g = len(means_by_tissue)
    mean_delta = float(means_by_tissue.mean()) if g else float("nan")
    sd_means = float(means_by_tissue.std(ddof=1)) if g > 1 else float("nan")
    se_means = sd_means / np.sqrt(g) if g > 1 else float("nan")
    # one-sample t, H1: mean > 0
    t_stat = mean_delta / se_means if (se_means and se_means > 0) else float("nan")
    # p-value from Student's t with df=g-1
    p_mix = float(1 - stats.t.cdf(t_stat, df=g - 1)) if g > 1 and np.isfinite(t_stat) else float("nan")
    # 95% CI on mean of tissue means
    tcrit = stats.t.ppf(0.975, df=g - 1) if g > 1 else float("nan")
    ci_mix = [mean_delta - tcrit * se_means if g > 1 else float("nan"), mean_delta + tcrit * se_means if g > 1 else float("nan")]

    # Secondary metrics and BH-FDR
    pvals = []
    metrics = ["PQ", "Dice", "AJI", "F1"]
    for m in metrics:
        d = wide[(m, "Delta")].to_numpy()
        p2 = stats.wilcoxon(d, alternative="two-sided", mode="auto").pvalue
        p1 = p2 / 2 if float(np.mean(d)) > 0 else 1 - p2 / 2
        pvals.append(p1)
    p_bh = _bh_adjust(pvals)
    rej = [pb < 0.05 for pb in p_bh]

    # Cohen's dz for PQ
    s_delta = float(np.std(d_pq, ddof=1) if np.std(d_pq, ddof=1) > 0 else np.inf)
    dz = float(np.mean(d_pq) / s_delta) if s_delta not in (0.0, np.inf) else float("inf")
    dz_ci = bootstrap_ci(d_pq, lambda x: float(np.mean(x) / (np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else np.inf)))

    # MDES at 80% power, one-sided alpha=0.05 (paired -> one-sample t on diffs)
    # MDES via normal approximation for one-sided test: (z_alpha + z_power)/sqrt(n)
    z_alpha = stats.norm.ppf(0.95)
    z_power = stats.norm.ppf(0.8)
    mdes_dz = float((z_alpha + z_power) / np.sqrt(n_overall)) if n_overall > 0 else float("nan")
    mdes_metric = float(mdes_dz * s_delta) if np.isfinite(mdes_dz) and np.isfinite(s_delta) else float("nan")

    # Save BH table
    bh_df = pd.DataFrame({"metric": metrics, "p_one_sided": pvals, "p_bh": p_bh, "reject@0.05": rej})
    os.makedirs(os.path.dirname(args.out_bh), exist_ok=True)
    bh_df.to_csv(args.out_bh, index=False)

    # Save summary
    summary = pd.DataFrame([
        {
            "n_overall": n_overall,
            "num_tissues": n_tissues,
            "median_n_per_tissue": median_nt,
            "min_n_per_tissue": range_nt[0],
            "max_n_per_tissue": range_nt[1],
            "wilcoxon_stat": float(wstat),
            "wilcoxon_p_one_sided": float(p_one),
            "median_delta_PQ": median_delta,
            "median_delta_PQ_ci_low": ci_low,
            "median_delta_PQ_ci_high": ci_high,
            "rank_biserial": rb,
            "cohens_dz": dz,
            "cohens_dz_ci_low": dz_ci[0],
            "cohens_dz_ci_high": dz_ci[1],
            "mixed_effects_mean_delta": mean_delta,
            "mixed_effects_ci_low": float(ci_mix[0]),
            "mixed_effects_ci_high": float(ci_mix[1]),
            "mixed_effects_p": p_mix,
            "mdes_dz_80power_one_sided": mdes_dz,
            "mdes_metric_units": mdes_metric,
        }
    ])
    summary.to_csv(args.out_summary, index=False)
    print(f"Saved summary to {args.out_summary} and BH table to {args.out_bh}")

    # Visuals
    import matplotlib.pyplot as plt
    import seaborn as sns
    os.makedirs(args.out_fig_dir, exist_ok=True)

    # Histogram/violin of Δ_PQ across all images
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.histplot(wide[("PQ", "Delta")].to_numpy(), kde=True, ax=ax1)
    ax1.set_title("Histogram of Δ_PQ (expert - unified)")
    ax1.set_xlabel("Δ_PQ")
    fig1.tight_layout()
    fig1.savefig(os.path.join(args.out_fig_dir, "hist_delta_pq.png"), dpi=200)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.violinplot(y=wide[("PQ", "Delta")].to_numpy(), ax=ax2)
    ax2.set_title("Violin of Δ_PQ")
    ax2.set_ylabel("Δ_PQ")
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.out_fig_dir, "violin_delta_pq.png"), dpi=200)
    plt.close(fig2)

    # Per-tissue boxplots of Δ_PQ
    df_plot = wide.copy().reset_index()
    cols = df_plot.columns
    tissue_key = ("tissue", "") if ("tissue", "") in cols else "tissue"
    pqdelta_key = ("PQ", "Delta") if ("PQ", "Delta") in cols else "PQ_Delta"
    df_plot = df_plot[[tissue_key, pqdelta_key]].copy()
    df_plot.columns = ["tissue", "PQ_Delta"]
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df_plot, x="tissue", y="PQ_Delta", ax=ax3)
    sns.stripplot(data=df_plot, x="tissue", y="PQ_Delta", ax=ax3, size=2, color="black", alpha=0.4)
    ax3.set_title("Per-tissue Δ_PQ")
    ax3.set_xlabel("tissue")
    ax3.set_ylabel("Δ_PQ")
    fig3.tight_layout()
    fig3.savefig(os.path.join(args.out_fig_dir, "box_by_tissue_delta_pq.png"), dpi=200)
    plt.close(fig3)

    # Forest plot: tissue-wise mean Δ_PQ with 95% CI (bootstrap)
    rows = []
    for tissue in sorted(df_plot["tissue"].unique()):
        vals = df_plot[df_plot["tissue"] == tissue]["PQ_Delta"].to_numpy()
        mean = float(np.mean(vals)) if vals.size else float("nan")
        lo, hi = bootstrap_ci(vals, np.mean) if vals.size else (float("nan"), float("nan"))
        rows.append({"tissue": tissue, "mean_delta": mean, "ci_low": lo, "ci_high": hi, "n": int(vals.size)})
    forest_df = pd.DataFrame(rows)
    forest_df.to_csv(args.out_forest_csv, index=False)

    # Plot forest
    fig4, ax4 = plt.subplots(figsize=(6, 0.4 * max(1, len(rows))))
    y = np.arange(len(rows))
    means = forest_df["mean_delta"].to_numpy()
    lo = forest_df["ci_low"].to_numpy()
    hi = forest_df["ci_high"].to_numpy()
    ax4.hlines(y=y, xmin=lo, xmax=hi, color='gray')
    ax4.plot(means, y, 'o', color='blue')
    ax4.axvline(0.0, color='red', linestyle='--', linewidth=1)
    ax4.set_yticks(y)
    ax4.set_yticklabels(forest_df["tissue"].tolist())
    ax4.set_xlabel("Mean Δ_PQ (expert - unified)")
    ax4.set_title("Tissue-wise Δ_PQ with 95% CI")
    fig4.tight_layout()
    fig4.savefig(os.path.join(args.out_fig_dir, "forest_delta_pq.png"), dpi=200)
    plt.close(fig4)

    # Paste-ready report sentences
    overall_n = n_overall
    T = n_tissues
    m = median_nt
    rmin, rmax = range_nt
    direction = "improvement" if median_delta > 0 else "no improvement"
    report_lines = []
    report_lines.append(
        f"Analyses were performed at the per-image level. A total of n={overall_n} patches had paired predictions across {T} tissues (median {m} per tissue; range {rmin}-{rmax}).\n"
    )
    report_lines.append(
        f"For PQ, a one-sided Wilcoxon signed-rank test on paired differences (expert − unified) found {direction} with median Δ={median_delta:.4f}, rank-biserial={rb:.3f}, p_BH={float(p_bh[0]):.4g}. "
        f"A mixed-effects model with tissue as a random intercept estimated mean Δ={mean_delta:.4f} (95% CI {ci_mix[0]:.4f}, {ci_mix[1]:.4f}), p={p_mix:.4g}.\n"
    )
    report_lines.append(
        f"With the observed sample size, the minimum detectable effect at 80% power (one-sided α=0.05) corresponds to d_z≈{mdes_dz:.3f} (≈{mdes_metric:.4f} PQ units).\n"
    )
    with open(args.out_report_txt, "w") as f:
        f.writelines(report_lines)


if __name__ == "__main__":
    main()


