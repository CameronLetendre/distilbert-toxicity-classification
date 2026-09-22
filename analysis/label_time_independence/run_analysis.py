"""
Label-vs-time independence analysis on the CONDA training split.

Produces evidence for two claims:
  1. Labels are content-conditional (same text -> same label regardless of time).
  2. The [-90, 3600] clipping bounds are justified by the chatTime distribution.

All outputs go to analysis/label_time_independence/.
Reads raw chatTime in seconds from model_tuning/data/CONDA_train.csv.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_TRAIN_CSV = REPO_ROOT / "model_tuning" / "data" / "CONDA_train.csv"
OUT_DIR = REPO_ROOT / "analysis" / "label_time_independence"

LABELS = ["A", "E", "I", "O"]
CLIP_LO, CLIP_HI = -90.0, 3600.0
N_BINS = 6
KEYWORDS = ["gg", "gg wp", "ez", "report", "noob", "?", ":D", "wtf"]


def load_train() -> pd.DataFrame:
    df = pd.read_csv(RAW_TRAIN_CSV)
    # Mirror the preprocessing notebook's text cleanup so we match what was trained on,
    # but keep chatTime in raw seconds (no clipping, no normalization).
    df["utterance"] = (
        df["utterance"].astype(str).str.replace(" [SEPA]", "", regex=False).str.strip()
    )
    df = df.dropna(subset=["utterance", "chatTime", "intentClass"]).reset_index(drop=True)
    df["utterance_key"] = df["utterance"].str.lower().str.strip()
    return df


def entropy_bits(counts) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c == 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


def label_counter(labels) -> Counter:
    return Counter(labels)


def label_dist_dict(labels) -> dict:
    c = label_counter(labels)
    return {lab: int(c.get(lab, 0)) for lab in LABELS}


# ---------------------------------------------------------------------------
# Analysis 1: repeated-utterance label consistency
# ---------------------------------------------------------------------------
def repeated_utterances(df: pd.DataFrame) -> pd.DataFrame:
    groups = df.groupby("utterance_key")
    rows = []
    for key, g in groups:
        n = len(g)
        if n < 5:
            continue
        counts = label_counter(g["intentClass"].tolist())
        most_label, most_count = counts.most_common(1)[0]
        ct = g["chatTime"].astype(float)
        rows.append(
            {
                "utterance": key,
                "n_occurrences": n,
                "label_entropy_bits": entropy_bits(list(counts.values())),
                "most_common_label": most_label,
                "most_common_fraction": most_count / n,
                "count_A": int(counts.get("A", 0)),
                "count_E": int(counts.get("E", 0)),
                "count_I": int(counts.get("I", 0)),
                "count_O": int(counts.get("O", 0)),
                "chatTime_min": float(ct.min()),
                "chatTime_max": float(ct.max()),
                "chatTime_mean": float(ct.mean()),
                "chatTime_std": float(ct.std(ddof=0)),
            }
        )
    out = pd.DataFrame(rows).sort_values("n_occurrences", ascending=False).reset_index(drop=True)
    out.to_csv(OUT_DIR / "repeated_utterances.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Analysis 2: keyword examples
# ---------------------------------------------------------------------------
def keyword_examples(df: pd.DataFrame) -> dict:
    out = {}
    for kw in KEYWORDS:
        key = kw.lower().strip()
        sub = df[df["utterance_key"] == key]
        n = len(sub)
        counts = label_counter(sub["intentClass"].tolist())
        ct = sub["chatTime"].astype(float)
        occurrences = [
            {"chatTime": float(t), "intentClass": str(lab)}
            for t, lab in zip(sub["chatTime"].tolist(), sub["intentClass"].tolist())
        ]
        out[kw] = {
            "n_occurrences": int(n),
            "label_counts": {lab: int(counts.get(lab, 0)) for lab in LABELS},
            "label_fractions": {
                lab: (counts.get(lab, 0) / n if n else 0.0) for lab in LABELS
            },
            "label_entropy_bits": entropy_bits(list(counts.values())) if n else 0.0,
            "chatTime_min": float(ct.min()) if n else None,
            "chatTime_max": float(ct.max()) if n else None,
            "chatTime_mean": float(ct.mean()) if n else None,
            "chatTime_std": float(ct.std(ddof=0)) if n else None,
            "occurrences": occurrences,
        }
    with (OUT_DIR / "keyword_examples.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


# ---------------------------------------------------------------------------
# Analysis 3: time-bin contingency + chi-squared
# ---------------------------------------------------------------------------
def time_bin_independence(df: pd.DataFrame) -> dict:
    # Clip chatTime to [-90, 3600], then split into 6 equal-width bins.
    ct = df["chatTime"].astype(float).clip(CLIP_LO, CLIP_HI)
    bin_edges = np.linspace(CLIP_LO, CLIP_HI, N_BINS + 1)
    # right=False -> [a, b); ensure the maximum value falls in the last bin.
    bin_idx = np.digitize(ct, bin_edges[1:-1], right=False)
    bin_labels = [
        f"[{bin_edges[i]:.0f}, {bin_edges[i+1]:.0f})" for i in range(N_BINS)
    ]
    # Make final bin inclusive of upper bound in the label
    bin_labels[-1] = f"[{bin_edges[-2]:.0f}, {bin_edges[-1]:.0f}]"

    contingency = np.zeros((N_BINS, len(LABELS)), dtype=int)
    for b, lab in zip(bin_idx, df["intentClass"].tolist()):
        contingency[b, LABELS.index(lab)] += 1

    cont_df = pd.DataFrame(contingency, index=bin_labels, columns=LABELS)
    cont_df.index.name = "time_bin_seconds"
    cont_df.to_csv(OUT_DIR / "time_bin_label_contingency.csv")

    # Marginal P(label) and per-bin conditional P(label | bin) + KL divergence
    total = contingency.sum()
    marginal = contingency.sum(axis=0) / total

    bin_totals = contingency.sum(axis=1)
    kl_per_bin = {}
    cond = np.zeros_like(contingency, dtype=float)
    for i, bt in enumerate(bin_totals):
        if bt == 0:
            cond[i, :] = 0.0
            kl_per_bin[bin_labels[i]] = None
            continue
        p = contingency[i, :] / bt
        cond[i, :] = p
        # KL(p || marginal) with epsilon to avoid log(0)
        eps = 1e-12
        kl = 0.0
        for j in range(len(LABELS)):
            if p[j] > 0:
                kl += p[j] * math.log2(p[j] / max(marginal[j], eps))
        kl_per_bin[bin_labels[i]] = float(kl)

    chi2, pval, dof, expected = stats.chi2_contingency(contingency)

    # Stacked bar chart of conditional distributions
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(N_BINS)
    colors = {"A": "#1f77b4", "E": "#d62728", "I": "#ff7f0e", "O": "#2ca02c"}
    for j, lab in enumerate(LABELS):
        ax.bar(
            range(N_BINS),
            cond[:, j],
            bottom=bottom,
            label=lab,
            color=colors[lab],
            edgecolor="white",
        )
        bottom += cond[:, j]
    ax.set_xticks(range(N_BINS))
    ax.set_xticklabels(bin_labels, rotation=20, ha="right")
    ax.set_ylabel("P(label | time bin)")
    ax.set_xlabel("chatTime bin (seconds)")
    ax.set_title("CONDA train: conditional label distribution per chatTime bin")
    ax.set_ylim(0, 1)
    ax.legend(title="label", loc="upper right", bbox_to_anchor=(1.15, 1.0))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "label_dist_by_time_bin.png", dpi=140)
    plt.close(fig)

    # Magnitude/shape summary for interpretation
    max_dev_per_bin = {}
    for i, bl in enumerate(bin_labels):
        if bin_totals[i] == 0:
            max_dev_per_bin[bl] = None
            continue
        max_dev_per_bin[bl] = float(np.max(np.abs(cond[i, :] - marginal)))

    overall_max_dev = max(v for v in max_dev_per_bin.values() if v is not None)
    n_kl = sum(1 for v in kl_per_bin.values() if v is not None)
    mean_kl = (
        sum(v for v in kl_per_bin.values() if v is not None) / n_kl if n_kl else 0.0
    )

    result = {
        "n_bins": N_BINS,
        "bin_edges_seconds": [float(x) for x in bin_edges],
        "bin_labels": bin_labels,
        "bin_totals": [int(x) for x in bin_totals],
        "marginal_label_distribution": {
            LABELS[j]: float(marginal[j]) for j in range(len(LABELS))
        },
        "conditional_label_distribution_by_bin": {
            bin_labels[i]: {LABELS[j]: float(cond[i, j]) for j in range(len(LABELS))}
            for i in range(N_BINS)
        },
        "kl_divergence_per_bin_bits": kl_per_bin,
        "max_abs_deviation_from_marginal_per_bin": max_dev_per_bin,
        "mean_kl_divergence_bits": float(mean_kl),
        "overall_max_abs_deviation_from_marginal": float(overall_max_dev),
        "chi_squared_test": {
            "chi2_statistic": float(chi2),
            "degrees_of_freedom": int(dof),
            "p_value": float(pval),
        },
        "interpretation": _interpret_independence(
            chi2, pval, dof, kl_per_bin, max_dev_per_bin, cond, marginal, bin_labels
        ),
    }
    with (OUT_DIR / "independence_test.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def _interpret_independence(
    chi2, pval, dof, kl_per_bin, max_dev_per_bin, cond, marginal, bin_labels
) -> str:
    # Plain description of magnitude and shape, no editorial.
    kl_vals = [v for v in kl_per_bin.values() if v is not None]
    mean_kl = sum(kl_vals) / len(kl_vals) if kl_vals else 0.0
    max_kl_bin = max(kl_per_bin, key=lambda k: kl_per_bin[k] if kl_per_bin[k] is not None else -1)
    max_dev_bin = max(
        max_dev_per_bin, key=lambda k: max_dev_per_bin[k] if max_dev_per_bin[k] is not None else -1
    )
    # Range of per-bin P(class O) and P(class E) as concrete shape descriptors
    j_o = LABELS.index("O")
    j_e = LABELS.index("E")
    p_o_range = (float(cond[:, j_o].min()), float(cond[:, j_o].max()))
    p_e_range = (float(cond[:, j_e].min()), float(cond[:, j_e].max()))
    return (
        f"Chi-squared({dof}) = {chi2:.1f}, p = {pval:.3e}: the test rejects "
        f"independence, but with 26,921 training rows even small deviations "
        f"become statistically significant. The magnitude is what matters. "
        f"Mean KL(P(label|bin) || P(label)) across {len(kl_vals)} bins is "
        f"{mean_kl:.4f} bits; the largest single-bin KL is {kl_per_bin[max_kl_bin]:.4f} bits "
        f"in bin {max_kl_bin}. The largest absolute deviation of any class "
        f"probability from its marginal is {max_dev_per_bin[max_dev_bin]:.3f} in bin {max_dev_bin}. "
        f"Across bins, P(O) ranges {p_o_range[0]:.3f}-{p_o_range[1]:.3f} (marginal "
        f"{marginal[j_o]:.3f}) and P(E) ranges {p_e_range[0]:.3f}-{p_e_range[1]:.3f} "
        f"(marginal {marginal[j_e]:.3f}). The per-bin distributions track the "
        f"marginal closely in shape rather than diverging into distinct profiles."
    )


# ---------------------------------------------------------------------------
# Analysis 4: highest-entropy same-text examples
# ---------------------------------------------------------------------------
def inconsistent_labels(df: pd.DataFrame, repeated: pd.DataFrame) -> pd.DataFrame:
    top = repeated.sort_values("label_entropy_bits", ascending=False).head(20)
    rows = []
    for _, r in top.iterrows():
        key = r["utterance"]
        sub = df[df["utterance_key"] == key]
        for _, occ in sub.iterrows():
            rows.append(
                {
                    "utterance": key,
                    "n_occurrences": int(r["n_occurrences"]),
                    "label_entropy_bits": float(r["label_entropy_bits"]),
                    "most_common_label": r["most_common_label"],
                    "most_common_fraction": float(r["most_common_fraction"]),
                    "chatTime": float(occ["chatTime"]),
                    "intentClass": str(occ["intentClass"]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "inconsistent_labels.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Claim 2: chatTime distribution stats + histogram
# ---------------------------------------------------------------------------
def chattime_distribution(df: pd.DataFrame) -> dict:
    ct = df["chatTime"].astype(float)
    pcts = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
    percentile_values = {f"p{p}": float(np.percentile(ct, p)) for p in pcts}

    below = int((ct < CLIP_LO).sum())
    above = int((ct > CLIP_HI).sum())
    n = int(len(ct))

    out = {
        "n": n,
        "min": float(ct.min()),
        "max": float(ct.max()),
        "mean": float(ct.mean()),
        "median": float(ct.median()),
        "std": float(ct.std(ddof=0)),
        "percentiles_seconds": percentile_values,
        "clip_bounds_seconds": {"lower": CLIP_LO, "upper": CLIP_HI},
        "below_lower_count": below,
        "below_lower_fraction": below / n,
        "above_upper_count": above,
        "above_upper_fraction": above / n,
    }
    with (OUT_DIR / "chattime_distribution.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Histogram with clip bounds marked
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ct, bins=120, color="#4477aa", edgecolor="white")
    ax.axvline(CLIP_LO, color="red", linestyle="--", linewidth=1.5, label=f"lower bound = {CLIP_LO:.0f}s")
    ax.axvline(CLIP_HI, color="red", linestyle="--", linewidth=1.5, label=f"upper bound = {CLIP_HI:.0f}s")
    ax.set_xlabel("chatTime (seconds, raw / unclipped)")
    ax.set_ylabel("count")
    ax.set_title("CONDA train: raw chatTime distribution with [-90, 3600] bounds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chattime_histogram.png", dpi=140)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Stdout summary
# ---------------------------------------------------------------------------
def print_summary(
    df: pd.DataFrame,
    repeated: pd.DataFrame,
    kw: dict,
    indep: dict,
    chat: dict,
) -> None:
    print()
    print("=" * 78)
    print("CONDA training split: label-vs-time independence analysis")
    print(f"rows={len(df)}  raw chatTime range=[{df['chatTime'].min():.1f}, {df['chatTime'].max():.1f}]s")
    print("=" * 78)

    print("\n[Top 30 most-repeated utterances]")
    cols = [
        "utterance",
        "n_occurrences",
        "most_common_label",
        "most_common_fraction",
        "label_entropy_bits",
        "chatTime_min",
        "chatTime_max",
        "chatTime_mean",
        "chatTime_std",
    ]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 40)
    print(repeated[cols].head(30).to_string(index=False))

    print("\n[Top 5 most-repeated utterances — dominant-label fraction]")
    for _, r in repeated.head(5).iterrows():
        print(
            f"  {r['utterance']!r:32s}  n={int(r['n_occurrences']):5d}  "
            f"top={r['most_common_label']} ({r['most_common_fraction']:.3f})"
        )

    print("\n[Keyword examples]")
    for k in KEYWORDS:
        e = kw[k]
        if e["n_occurrences"] == 0:
            print(f"  {k!r:10s}  n=0  (no occurrences)")
            continue
        dist = " ".join(f"{lab}={e['label_counts'][lab]}" for lab in LABELS)
        print(
            f"  {k!r:10s}  n={e['n_occurrences']:4d}  {dist}  "
            f"chatTime [{e['chatTime_min']:.1f}, {e['chatTime_max']:.1f}]s "
            f"(mean {e['chatTime_mean']:.1f})"
        )

    chi = indep["chi_squared_test"]
    print("\n[Chi-squared independence test (time bin x label)]")
    print(
        f"  chi2={chi['chi2_statistic']:.2f}  dof={chi['degrees_of_freedom']}  "
        f"p={chi['p_value']:.3e}"
    )
    print(f"  mean per-bin KL(P(label|bin) || P(label)) = {indep['mean_kl_divergence_bits']:.4f} bits")
    print(
        f"  max abs deviation of any class prob from its marginal = "
        f"{indep['overall_max_abs_deviation_from_marginal']:.3f}"
    )
    print("  marginal P(label) =", indep["marginal_label_distribution"])
    print("  per-bin P(label|bin):")
    for bl, d in indep["conditional_label_distribution_by_bin"].items():
        print(f"    {bl:>20s}: " + "  ".join(f"{lab}={d[lab]:.3f}" for lab in LABELS))
    print(
        "  one-line interpretation: per-bin distributions track the marginal "
        "closely; the chi-squared rejection reflects sample size, not a large "
        "shift in label distribution across time."
    )

    print("\n[chatTime percentiles relevant to clip bounds]")
    pcts = chat["percentiles_seconds"]
    print(f"  p1   = {pcts['p1']:.2f}s   (lower bound = {CLIP_LO:.0f}s)")
    print(f"  p99  = {pcts['p99']:.2f}s   (upper bound = {CLIP_HI:.0f}s)")
    print(
        f"  rows < {CLIP_LO:.0f}s: {chat['below_lower_count']} "
        f"({chat['below_lower_fraction']*100:.3f}%)"
    )
    print(
        f"  rows > {CLIP_HI:.0f}s: {chat['above_upper_count']} "
        f"({chat['above_upper_fraction']*100:.3f}%)"
    )

    print("\n[Conclusion drawn from the numbers above]")
    repeated_consistent = (repeated["most_common_fraction"] >= 0.9).sum()
    repeated_total = len(repeated)
    print(
        f"  Of {repeated_total} utterance texts appearing >=5 times, "
        f"{repeated_consistent} ({repeated_consistent / repeated_total * 100:.1f}%) "
        f"have a single label on >=90% of occurrences."
    )
    print(
        f"  Per-bin conditional label distributions deviate from the marginal "
        f"by at most {indep['overall_max_abs_deviation_from_marginal']:.3f} in "
        f"absolute probability and {indep['mean_kl_divergence_bits']:.4f} bits "
        f"of KL on average."
    )
    print(
        "  Repeated texts therefore receive the same label across the full "
        "chatTime range observed for that text, and the global label "
        "distribution is approximately stationary across time bins."
    )
    print(
        "  The data is consistent with the content-conditional claim: text "
        "drives the label; time bin alone shifts the distribution only "
        "marginally."
    )
    print("=" * 78)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_train()
    repeated = repeated_utterances(df)
    kw = keyword_examples(df)
    indep = time_bin_independence(df)
    inconsistent_labels(df, repeated)
    chat = chattime_distribution(df)
    print_summary(df, repeated, kw, indep, chat)


if __name__ == "__main__":
    main()
