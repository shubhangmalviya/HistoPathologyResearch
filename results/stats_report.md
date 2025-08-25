# Expert vs Unified Statistical Comparison
## Tests
- Wilcoxon signed-rank (one-sided, expert > unified) with Benjamini–Hochberg correction across metrics.
- Paired t-test (one-sided, expert > unified) with Benjamini–Hochberg correction, reported as sensitivity analysis.
- Effect sizes: rank-biserial correlation (Wilcoxon) and Cohen's dz (paired).

### Wilcoxon signed-rank results
| metric | statistic | p_value | effect_r_rank_biserial | p_adj_bh | significant@0.05 |
|---|---|---|---|---|---|
| Dice | 0.0 | 1.0 | -1.0 | 1.2916666666666667 | False |
| AJI | 2.0 | 0.9375 | -0.7333333333333333 | 1.9375 | False |
| PQ | 1.0 | 0.96875 | -0.8666666666666667 | 3.75 | False |
| F1 | 1.0 | 0.96875 | -0.8666666666666667 | 1.0 | False |

### Paired t-test results
| metric | statistic | p_value | effect_cohens_dz | p_adj_bh | significant@0.05 |
|---|---|---|---|---|---|
| Dice | -7.260518481363516 | 0.9990445006973525 | -3.2470025752444727 | 1.2720303014750607 | False |
| AJI | -1.6816948071826885 | 0.9160396680043502 | -0.7520767812537785 | 3.664158672017401 | False |
| PQ | -2.2067870871972874 | 0.9540227261062956 | -0.9869051877683781 | 1.899783063444916 | False |
| F1 | -2.1299199249740877 | 0.949891531722458 | -0.9525291477746622 | 0.9990445006973525 | False |

