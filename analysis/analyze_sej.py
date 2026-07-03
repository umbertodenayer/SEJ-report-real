#!/usr/bin/env python3
"""Reproduce the Classical Model analysis used in the WI4138 report.

The input CSV is never copied into the repository. Outputs identify respondents
only as Expert 1, Expert 2, etc. Run from the repository root with:

    python analysis/analyze_sej.py /path/to/formspree_submissions.csv
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "generated"
FIG = ROOT / "figures" / "generated"
PROBS = np.array([0.05, 0.45, 0.45, 0.05])
REALIZATIONS = np.array([
    480_000, 890.171, 87_211, 8_274_478, 42.4,
    34_579, 19.0, 91_660, 24.7, 23_600,
])

# Uniform backgrounds for proportions; log-uniform for positive scale variables.
CAL_LOG = np.array([True, True, True, True, False, True, False, True, False, True])
TGT_LOG = np.array([True, True, True, False])


@dataclass
class ItemRange:
    low: float
    high: float
    log: bool

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        return np.log(x) if self.log else x


def load(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No submissions found")
    cal = np.array([[[float(r[f"cal{j:02d}_p{q}"]) for q in ("05", "50", "95")]
                     for j in range(1, 11)] for r in rows])
    tgt = np.array([[[float(r[f"tgt{j}_p{q}"]) for q in ("05", "50", "95")]
                     for j in range(1, 5)] for r in rows])
    if np.any(np.diff(cal, axis=2) <= 0) or np.any(np.diff(tgt, axis=2) <= 0):
        raise ValueError("Every elicited 5th, 50th and 95th percentile must be strictly ordered")
    return rows, cal, tgt


def intrinsic_ranges(values: np.ndarray, log_flags: np.ndarray, realizations=None, overshoot=0.10):
    ranges = []
    for j, is_log in enumerate(log_flags):
        points = values[:, j, :].ravel()
        if realizations is not None:
            points = np.append(points, realizations[j])
        if is_log and np.min(points) <= 0:
            raise ValueError(f"Item {j + 1} is log-uniform but contains a non-positive value")
        transformed = np.log(points) if is_log else points
        width = transformed.max() - transformed.min()
        low_t = transformed.min() - overshoot * width
        high_t = transformed.max() + overshoot * width
        low, high = (math.exp(low_t), math.exp(high_t)) if is_log else (low_t, high_t)
        ranges.append(ItemRange(low, high, bool(is_log)))
    return ranges


def bin_counts(quantiles: np.ndarray, realizations: np.ndarray):
    counts = np.zeros(4, dtype=int)
    for q, x in zip(quantiles, realizations):
        counts[np.searchsorted(q, x, side="left")] += 1
    return counts


def calibration(counts: np.ndarray):
    n = counts.sum()
    observed = counts / n
    positive = observed > 0
    info = np.sum(observed[positive] * np.log(observed[positive] / PROBS[positive]))
    return float(chi2.sf(2 * n * info, df=3))


def information(quantiles: np.ndarray, item_range: ItemRange):
    edges = np.r_[item_range.low, quantiles, item_range.high]
    z = item_range.transform(edges)
    background_prob = np.diff(z) / (z[-1] - z[0])
    return float(np.sum(PROBS * np.log(PROBS / background_prob)))


def expert_metrics(cal: np.ndarray, ranges):
    counts = np.array([bin_counts(q, REALIZATIONS) for q in cal])
    calibrations = np.array([calibration(c) for c in counts])
    infos_by_item = np.array([[information(cal[e, j], ranges[j]) for j in range(10)]
                              for e in range(len(cal))])
    return counts, calibrations, infos_by_item, infos_by_item.mean(axis=1)


def cdf(x: float, q: np.ndarray, r: ItemRange):
    edges = np.r_[r.low, q, r.high]
    z, zx = r.transform(edges), float(r.transform([x])[0])
    if zx <= z[0]: return 0.0
    if zx >= z[-1]: return 1.0
    k = np.searchsorted(z, zx, side="right") - 1
    return float(PROBS[:k].sum() + PROBS[k] * (zx - z[k]) / (z[k + 1] - z[k]))


def mixture_quantiles(values: np.ndarray, weights: np.ndarray, ranges):
    result = np.zeros((values.shape[1], 3))
    for j, r in enumerate(ranges):
        def mix(x): return sum(weights[e] * cdf(x, values[e, j], r) for e in range(len(weights)))
        for k, p in enumerate((0.05, 0.50, 0.95)):
            result[j, k] = brentq(lambda x: mix(x) - p, r.low, r.high)
    return result


def dm_seed_score(seed_quantiles, ranges):
    counts = bin_counts(seed_quantiles, REALIZATIONS)
    cal = calibration(counts)
    info = np.mean([information(seed_quantiles[j], ranges[j]) for j in range(10)])
    return counts, cal, info, cal * info


def optimize(cal_values, info_values, cal_data, cal_ranges):
    candidates = sorted(set([0.0, *cal_values.tolist()]))
    options = []
    for alpha in candidates:
        raw = cal_values * info_values * (cal_values >= alpha)
        if raw.sum() == 0: continue
        weights = raw / raw.sum()
        dm_cal = mixture_quantiles(cal_data, weights, cal_ranges)
        counts, score_cal, score_info, combined = dm_seed_score(dm_cal, cal_ranges)
        options.append((combined, alpha, weights, dm_cal, counts, score_cal, score_info))
    return max(options, key=lambda x: (x[0], x[1]))


def evaluate_threshold(alpha, cal_values, info_values, cal_data, cal_ranges, tgt_data, tgt_ranges):
    raw = cal_values * info_values * (cal_values >= alpha)
    if raw.sum() == 0:
        raise ValueError(f"No expert survives alpha={alpha}")
    weights = raw / raw.sum()
    seed_quantiles = mixture_quantiles(cal_data, weights, cal_ranges)
    metrics = dm_seed_score(seed_quantiles, cal_ranges)
    targets = mixture_quantiles(tgt_data, weights, tgt_ranges)
    return weights, metrics, targets


def evaluate_configuration(cal, tgt, cal_log, tgt_log, overshoot):
    cal_ranges = intrinsic_ranges(cal, cal_log, REALIZATIONS, overshoot)
    tgt_ranges = intrinsic_ranges(tgt, tgt_log, overshoot=overshoot)
    counts, cal_scores, _, mean_infos = expert_metrics(cal, cal_ranges)
    combined, alpha, weights, _, dm_counts, dm_cal, dm_info = optimize(
        cal_scores, mean_infos, cal, cal_ranges)
    forecasts = mixture_quantiles(tgt, weights, tgt_ranges)
    return {
        "overshoot": overshoot,
        "background": "mixed" if np.any(cal_log) else "all-uniform",
        "alpha": alpha,
        "weights": weights,
        "calibration": dm_cal,
        "information": dm_info,
        "combined": combined,
        "targets": forecasts,
    }


def fmt(x, digits=3):
    return f"{x:.{digits}f}"


def write_tex(counts, cal_scores, mean_infos, weights, dm_metrics, forecasts,
              ewdm_metrics, ewdm_forecasts, fixed_metrics, alpha, sensitivity):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in range(len(cal_scores)):
        rows.append(
            f"Expert {e+1} & {'--'.join(map(str, counts[e]))} & {cal_scores[e]:.4f} & "
            f"{mean_infos[e]:.3f} & {cal_scores[e]*mean_infos[e]:.4f} & {weights[e]:.3f} \\\\"
        )
    (OUT / "expert_table.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
    _, dm_cal, dm_info, dm_combined = dm_metrics
    _, ew_cal, ew_info, ew_combined = ewdm_metrics
    _, fixed_cal, fixed_info, fixed_combined = fixed_metrics
    (OUT / "dm_table.tex").write_text(
        f"Equal-weight DM & {ew_cal:.4f} & {ew_info:.3f} & {ew_combined:.4f} \\\\\n"
        f"Performance-weight DM ($\\alpha=0.05$) & {fixed_cal:.4f} & {fixed_info:.3f} & {fixed_combined:.4f} \\\\\n"
        f"Optimized performance-weight DM & {dm_cal:.4f} & {dm_info:.3f} & {dm_combined:.4f} \\\\\n\\bottomrule\n"
    )
    labels = ["Completed dwellings", "Building permits", "Rotterdam transaction price", "Free-sector rent growth"]
    target_rows = []
    for j, label in enumerate(labels):
        q = forecasts[j]
        eq = ewdm_forecasts[j]
        decimals = 2 if j == 3 else 0
        target_rows.append(
            f"{label} & {eq[0]:,.{decimals}f} & {eq[1]:,.{decimals}f} & {eq[2]:,.{decimals}f} & "
            f"{q[0]:,.{decimals}f} & {q[1]:,.{decimals}f} & {q[2]:,.{decimals}f} \\\\"
        )
    (OUT / "target_table.tex").write_text("\n".join(target_rows) + "\n\\bottomrule\n")
    macros = [
        f"\\newcommand{{\\OptimalAlpha}}{{{alpha:.4f}}}",
        f"\\newcommand{{\\OptimizedDMCalibration}}{{{dm_cal:.4f}}}",
        f"\\newcommand{{\\OptimizedDMInformation}}{{{dm_info:.3f}}}",
    ]
    for j, stem in enumerate(("Dwellings", "Permits", "Price", "Rent")):
        decimals = 2 if j == 3 else 0
        for k, quantile in enumerate(("Five", "Median", "NinetyFive")):
            macros.append(f"\\newcommand{{\\{stem}{quantile}}}{{{forecasts[j,k]:,.{decimals}f}}}")
    (OUT / "results_macros.tex").write_text("\n".join(macros) + "\n")
    sensitivity_rows = []
    for s in sensitivity:
        w = s["weights"]
        q = s["targets"][0]
        label = (f"{int(s['overshoot']*100)}\\% overshoot"
                 if s["background"] == "mixed" else "All-uniform, 10\\%")
        sensitivity_rows.append(
            f"{label} & {s['alpha']:.4f} & {s['calibration']:.4f} & {s['information']:.3f} & "
            f"{s['combined']:.4f} & {w[1]+w[3]:.3f} & {q[1]:,.0f} \\\\"
        )
    (OUT / "sensitivity_table.tex").write_text(
        "\n".join(sensitivity_rows) + "\n\\bottomrule\n")


def range_plots(cal, ranges):
    FIG.mkdir(parents=True, exist_ok=True)
    for j in range(10):
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        y = np.arange(1, len(cal) + 1)
        ax.hlines(y, cal[:, j, 0], cal[:, j, 2], color="#00A6D2", lw=3)
        ax.scatter(cal[:, j, 1], y, color="#111827", s=22, zorder=3, label="Median")
        ax.axvline(REALIZATIONS[j], color="#D97706", lw=2, ls="--", label="Realization")
        ax.set_yticks(y, [f"Expert {i}" for i in y])
        ax.invert_yaxis(); ax.grid(axis="x", alpha=.2); ax.set_title(f"Calibration question {j+1}")
        ax.legend(frameon=False, ncol=2, loc="lower right"); fig.tight_layout()
        fig.savefig(FIG / f"calibration-{j+1:02d}.pdf", bbox_inches="tight")
        plt.close(fig)


def summary_plots(cal_scores, mean_infos, weights, forecasts, ewdm_forecasts, alpha):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sizes = 55 + 900 * weights
    ax.scatter(cal_scores, mean_infos, s=sizes, color="#00A6D2", edgecolor="#111827", alpha=.85)
    for i, (x, y) in enumerate(zip(cal_scores, mean_infos), 1):
        ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.axvline(alpha, color="#D97706", ls="--", lw=1.6,
               label=rf"Optimal threshold $\alpha^*={alpha:.4f}$")
    ax.set_xlabel("Calibration score")
    ax.set_ylabel("Mean information score")
    ax.set_title("Expert performance (marker area indicates optimized weight)")
    ax.grid(alpha=.2); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(FIG / "expert-performance.pdf", bbox_inches="tight")
    plt.close(fig)

    labels = ["Completed dwellings", "Building permits", "Rotterdam price", "Free-sector rent growth"]
    scales = [1_000, 1_000, 1_000, 1]
    units = ["thousands", "thousands", "EUR thousands", "%"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    for j, ax in enumerate(axes.flat):
        series = (("Equal weight", ewdm_forecasts[j], "#64748B"),
                  ("Optimized", forecasts[j], "#00A6D2"))
        for y, (name, q, color) in enumerate(series):
            q = q / scales[j]
            ax.hlines(y, q[0], q[2], color=color, lw=4)
            ax.scatter(q[1], y, color="#111827", s=24, zorder=3)
        ax.set_yticks([0, 1], ["Equal weight", "Optimized"])
        ax.set_title(labels[j], fontsize=10)
        ax.set_xlabel(units[j], fontsize=9)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Decision Maker target forecasts: medians and 90% intervals", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "target-forecasts.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python analysis/analyze_sej.py /path/to/formspree_submissions.csv")
    rows, cal, tgt = load(Path(sys.argv[1]))
    cal_ranges = intrinsic_ranges(cal, CAL_LOG, REALIZATIONS)
    tgt_ranges = intrinsic_ranges(tgt, TGT_LOG)
    counts, cal_scores, infos_by_item, mean_infos = expert_metrics(cal, cal_ranges)
    combined, alpha, weights, dm_seed, dm_counts, dm_cal, dm_info = optimize(
        cal_scores, mean_infos, cal, cal_ranges)
    forecasts = mixture_quantiles(tgt, weights, tgt_ranges)
    ew = np.ones(len(rows)) / len(rows)
    ew_seed = mixture_quantiles(cal, ew, cal_ranges)
    ew_metrics = dm_seed_score(ew_seed, cal_ranges)
    ew_forecasts = mixture_quantiles(tgt, ew, tgt_ranges)
    dm_metrics = (dm_counts, dm_cal, dm_info, combined)
    fixed_weights, fixed_metrics, fixed_forecasts = evaluate_threshold(
        0.05, cal_scores, mean_infos, cal, cal_ranges, tgt, tgt_ranges)
    sensitivity = [
        evaluate_configuration(cal, tgt, CAL_LOG, TGT_LOG, o) for o in (0.05, 0.10, 0.20)
    ]
    sensitivity.append(evaluate_configuration(
        cal, tgt, np.zeros(10, dtype=bool), np.zeros(4, dtype=bool), 0.10))
    write_tex(counts, cal_scores, mean_infos, weights, dm_metrics, forecasts,
              ew_metrics, ew_forecasts, fixed_metrics, alpha, sensitivity)
    range_plots(cal, cal_ranges)
    summary_plots(cal_scores, mean_infos, weights, forecasts, ew_forecasts, alpha)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_experts": len(rows), "optimal_alpha": alpha,
        "experts": [{"id": f"Expert {i+1}", "bins": counts[i].tolist(),
                     "calibration": cal_scores[i], "mean_information": mean_infos[i],
                     "optimized_weight": weights[i]} for i in range(len(rows))],
        "equal_weight_dm": {"calibration": ew_metrics[1], "information": ew_metrics[2],
                            "combined": ew_metrics[3], "targets": ew_forecasts.tolist()},
        "fixed_005_dm": {"calibration": fixed_metrics[1], "information": fixed_metrics[2],
                         "combined": fixed_metrics[3], "weights": fixed_weights.tolist(),
                         "targets": fixed_forecasts.tolist()},
        "optimized_dm": {"calibration": dm_cal, "information": dm_info,
                         "combined": combined, "targets": forecasts.tolist()},
        "sensitivity": [
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
            for s in sensitivity
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
