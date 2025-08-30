import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(pvals: List[float]) -> List[float]:
    m = len(pvals)
    ranks = np.argsort(np.argsort(pvals)) + 1
    pvals = np.array(pvals, dtype=float)
    adjusted = pvals * m / ranks
    # enforce monotonicity
    adjusted_sorted = np.minimum.accumulate(np.sort(adjusted)[::-1])[::-1]
    # map back to original order
    return adjusted_sorted[np.argsort(ranks)]


def rank_biserial_from_diffs(diffs: np.ndarray) -> float:
    # Remove zeros
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


def compare_metrics(df: pd.DataFrame, metrics: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_wilcoxon = []
    rows_ttest = []
    for metric in metrics:
        expert_vals = df[df["model"] == "expert"].sort_values("tissue")[metric].to_numpy()
        unified_vals = df[df["model"] == "unified"].sort_values("tissue")[metric].to_numpy()
        if expert_vals.shape != unified_vals.shape:
            raise ValueError("Mismatched expert/unified lengths")
        if expert_vals.size == 0:
            continue
        diffs = expert_vals - unified_vals

        # Wilcoxon signed-rank, one-sided (expert > unified)
        w_stat, w_p = stats.wilcoxon(expert_vals, unified_vals, alternative="greater", zero_method="pratt")
        r_rb = rank_biserial_from_diffs(diffs)

        # Paired t-test, one-sided (expert > unified)
        t_stat, t_p = stats.ttest_rel(expert_vals, unified_vals, alternative="greater")
        dz = float(np.mean(diffs) / (np.std(diffs, ddof=1) if np.std(diffs, ddof=1) > 0 else np.inf))

        rows_wilcoxon.append({"metric": metric, "statistic": float(w_stat), "p_value": float(w_p), "effect_r_rank_biserial": r_rb})
        rows_ttest.append({"metric": metric, "statistic": float(t_stat), "p_value": float(t_p), "effect_cohens_dz": dz})

    wilcox_df = pd.DataFrame(rows_wilcoxon)
    ttest_df = pd.DataFrame(rows_ttest)

    if not wilcox_df.empty:
        wilcox_df["p_adj_bh"] = benjamini_hochberg(wilcox_df["p_value"].tolist())
        wilcox_df["significant@0.05"] = wilcox_df["p_adj_bh"] < 0.05
    if not ttest_df.empty:
        ttest_df["p_adj_bh"] = benjamini_hochberg(ttest_df["p_value"].tolist())
        ttest_df["significant@0.05"] = ttest_df["p_adj_bh"] < 0.05

    return wilcox_df, ttest_df


def write_report_md(path: str, wilcox_df: pd.DataFrame, ttest_df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines: List[str] = []
    lines.append("# Expert vs Unified Statistical Comparison\n")
    lines.append("## Tests\n")
    lines.append("- Wilcoxon signed-rank (one-sided, expert > unified) with Benjamini–Hochberg correction across metrics.\n")
    lines.append("- Paired t-test (one-sided, expert > unified) with Benjamini–Hochberg correction, reported as sensitivity analysis.\n")
    lines.append("- Effect sizes: rank-biserial correlation (Wilcoxon) and Cohen's dz (paired).\n")
    lines.append("\n")

    def frame_to_md(df: pd.DataFrame, title: str) -> List[str]:
        if df.empty:
            return [f"### {title}\n", "No data.\n\n"]
        cols = df.columns.tolist()
        md = [f"### {title}\n", "| " + " | ".join(cols) + " |\n", "|" + "|".join(["---"] * len(cols)) + "|\n"]
        for _, r in df.iterrows():
            md.append("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
        md.append("\n")
        return md

    lines += frame_to_md(wilcox_df, "Wilcoxon signed-rank results")
    lines += frame_to_md(ttest_df, "Paired t-test results")

    with open(path, "w") as f:
        f.writelines(lines)


def main():
    from utils.paths import results_dir, artifacts_root
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", type=str, default=os.path.join(results_dir(), "metrics.csv"))
    parser.add_argument("--out_wilcox_csv", type=str, default=os.path.join(results_dir(), "wilcoxon.csv"))
    parser.add_argument("--out_ttest_csv", type=str, default=os.path.join(results_dir(), "ttest.csv"))
    parser.add_argument("--out_md", type=str, default=os.path.join(results_dir(), "stats_report.md"))
    args = parser.parse_args()

    df = pd.read_csv(args.metrics_csv)
    # ensure per-tissue pairing
    tissues = sorted(df["tissue"].unique().tolist())
    metrics = ["Dice", "AJI", "PQ", "F1"]
    wilcox_df, ttest_df = compare_metrics(df, metrics)

    os.makedirs(os.path.dirname(args.out_wilcox_csv), exist_ok=True)
    wilcox_df.to_csv(args.out_wilcox_csv, index=False)
    ttest_df.to_csv(args.out_ttest_csv, index=False)
    write_report_md(args.out_md, wilcox_df, ttest_df)
    print(f"Saved: {args.out_wilcox_csv}, {args.out_ttest_csv}, {args.out_md}")


if __name__ == "__main__":
    main()


