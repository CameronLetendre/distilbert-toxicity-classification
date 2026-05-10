"""Generate result graphs from evaluate.py / evaluate_no_time.py JSON outputs.

Produces:
  - per_class_f1.png : per-class F1 bar chart, with-time vs no-time
  - macro_summary.png : UCA + macro-F1 comparison
  - confusion_with_time.png, confusion_no_time.png : 4x4 heatmaps
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "src/model/results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

WITH_TIME = RESULTS_DIR / "metrics_with_time.json"
NO_TIME = RESULTS_DIR / "metrics_no_time.json"


def load(path):
    if not path.exists():
        print(f"[skip] {path.name} not found")
        return None
    with open(path) as f:
        return json.load(f)


def plot_per_class_f1(wt, nt):
    if not (wt and nt):
        return
    classes = wt["label_order"]
    wt_vals = [wt["per_class_f1"][c] for c in classes]
    nt_vals = [nt["per_class_f1"][c] for c in classes]

    x = np.arange(len(classes))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - width/2, nt_vals, width, label="no time token", color="#94a3b8")
    b2 = ax.bar(x + width/2, wt_vals, width, label="with [TIME]", color="#38bdf8")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{label_name(c)}" for c in classes])
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1: time-token ablation")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                    f"{h:.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    out = PLOTS_DIR / "per_class_f1.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def plot_macro_summary(wt, nt):
    if not (wt and nt):
        return
    metrics = ["UCA", "Macro F1"]
    nt_vals = [nt["uca"], nt["macro_f1"]]
    wt_vals = [wt["uca"], wt["macro_f1"]]

    x = np.arange(len(metrics))
    width = 0.38
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(x - width/2, nt_vals, width, label="no time token", color="#94a3b8")
    ax.bar(x + width/2, wt_vals, width, label="with [TIME]", color="#38bdf8")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Overall metrics")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()

    for i, (a, b) in enumerate(zip(nt_vals, wt_vals)):
        ax.text(i - width/2, a + 0.015, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + width/2, b + 0.015, f"{b:.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    out = PLOTS_DIR / "macro_summary.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def plot_confusion(metrics, fname):
    if metrics is None:
        return
    cm = np.array(metrics["confusion_matrix"], dtype=float)
    labels = metrics["label_order"]
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix — {metrics['run']}\n(row-normalized; counts shown)")

    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{int(cm[i, j])}\n{cm_norm[i, j]*100:.1f}%",
                    ha="center", va="center", color=color, fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = PLOTS_DIR / fname
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def label_name(c):
    return {"E": "Explicit", "I": "Implicit", "A": "Action", "O": "Other"}[c]


def write_results_table(wt, nt):
    if not (wt and nt):
        return
    classes = wt["label_order"]
    rows = [("UCA", nt["uca"], wt["uca"]),
            ("Macro F1", nt["macro_f1"], wt["macro_f1"])]
    for c in classes:
        rows.append((f"F1[{c}]",
                     nt["per_class_f1"][c],
                     wt["per_class_f1"][c]))

    csv_path = RESULTS_DIR / "results_table.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("metric,no_time,with_time,delta\n")
        for name, n, w in rows:
            f.write(f"{name},{n:.4f},{w:.4f},{w - n:+.4f}\n")
    print(f"wrote {csv_path}")

    md_path = RESULTS_DIR / "results_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Metric | no time | with [TIME] | Δ |\n")
        f.write("|---|---:|---:|---:|\n")
        for name, n, w in rows:
            f.write(f"| {name} | {n:.4f} | {w:.4f} | {w - n:+.4f} |\n")
    print(f"wrote {md_path}")

    print("\n=== Results Table ===")
    print(f"{'Metric':<10} {'no_time':>10} {'with_time':>10} {'delta':>10}")
    for name, n, w in rows:
        print(f"{name:<10} {n:>10.4f} {w:>10.4f} {w - n:>+10.4f}")


def render_results_table_png(wt, nt):
    """Render the comparison table as a standalone PNG for slides."""
    if not (wt and nt):
        return
    classes = wt["label_order"]
    rows = [("UCA", nt["uca"], wt["uca"]),
            ("Macro F1", nt["macro_f1"], wt["macro_f1"])]
    for c in classes:
        rows.append((f"F1[{c}]",
                     nt["per_class_f1"][c],
                     wt["per_class_f1"][c]))

    cell_text = [[name, f"{n:.4f}", f"{w:.4f}", f"{w - n:+.4f}"]
                 for name, n, w in rows]
    col_labels = ["Metric", "no time", "with [TIME]", "Δ"]

    fig, ax = plt.subplots(figsize=(7.5, 0.45 * (len(rows) + 1) + 0.6))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   cellLoc="center", colLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.4)

    # Header styling
    for j in range(len(col_labels)):
        c = tbl[(0, j)]
        c.set_facecolor("#1e293b")
        c.set_text_props(color="white", fontweight="bold")

    # Highlight delta column by sign
    for i, (_, n, w) in enumerate(rows, start=1):
        delta = w - n
        cell = tbl[(i, 3)]
        if delta > 0:
            cell.set_facecolor("#dcfce7")
        elif delta < 0:
            cell.set_facecolor("#fee2e2")

    ax.set_title("Time-token ablation — full results",
                 fontsize=13, fontweight="bold", pad=12)
    out = PLOTS_DIR / "results_table.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    wt = load(WITH_TIME)
    nt = load(NO_TIME)
    plot_per_class_f1(wt, nt)
    plot_macro_summary(wt, nt)
    plot_confusion(wt, "confusion_with_time.png")
    plot_confusion(nt, "confusion_no_time.png")
    write_results_table(wt, nt)
    render_results_table_png(wt, nt)


if __name__ == "__main__":
    main()
