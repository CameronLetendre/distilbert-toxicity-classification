"""Generate presentation-ready charts for the Data + Preprocessing slide."""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT_DIR = Path(__file__).parent / "slide_assets"
OUT_DIR.mkdir(exist_ok=True)

# Palette — clean, presentation-friendly
BG = "#FFFFFF"
INK = "#1F2937"
MUTED = "#6B7280"
ACCENT = "#2563EB"
COLORS = {
    "O": "#9CA3AF",  # neutral gray (majority)
    "E": "#EF4444",  # red (explicit)
    "I": "#F59E0B",  # amber (implicit)
    "A": "#10B981",  # green (action)
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def chart_class_distribution():
    """Donut chart of intent class distribution."""
    labels = ["O — Other", "E — Explicit", "A — Action", "I — Implicit"]
    keys = ["O", "E", "A", "I"]
    counts = [19982, 3528, 1719, 1692]
    total = sum(counts)
    colors = [COLORS[k] for k in keys]

    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=200)
    fig.patch.set_facecolor(BG)

    wedges, _ = ax.pie(
        counts,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor=BG, linewidth=3),
    )

    ax.text(0, 0.08, f"{total:,}", ha="center", va="center",
            fontsize=28, fontweight="bold", color=INK)
    ax.text(0, -0.12, "utterances", ha="center", va="center",
            fontsize=12, color=MUTED)

    legend_labels = [
        f"{lbl}   {c:,}  ({c / total * 100:.1f}%)"
        for lbl, c in zip(labels, counts)
    ]
    ax.legend(
        wedges, legend_labels,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        frameon=False,
        fontsize=13,
        handlelength=1.2,
        handleheight=1.2,
    )

    ax.set_title("Intent Class Distribution — Training Set",
                 fontsize=16, fontweight="bold", color=INK, pad=18)
    plt.tight_layout()
    out = OUT_DIR / "01_class_distribution.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved {out}")


def chart_columns_reduction():
    """Visual showing 10 raw columns dropped to 3 kept."""
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    raw_cols = [
        "Id", "conversationId", "playerId", "matchId",
        "playerSlot", "slotClasses", "slotTokens",
        "utterance", "chatTime", "intentClass",
    ]
    kept = {"utterance", "chatTime", "intentClass"}

    # Left panel: raw
    ax.text(2.0, 9.2, "RAW", fontsize=13, fontweight="bold",
            color=MUTED, ha="center")
    ax.text(2.0, 8.7, "10 columns", fontsize=11, color=MUTED, ha="center")

    for i, col in enumerate(raw_cols):
        y = 8.0 - i * 0.62
        is_kept = col in kept
        face = "#DBEAFE" if is_kept else "#F3F4F6"
        edge = ACCENT if is_kept else "#D1D5DB"
        text_color = INK if is_kept else "#9CA3AF"
        weight = "bold" if is_kept else "normal"

        box = FancyBboxPatch(
            (0.6, y - 0.22), 2.8, 0.44,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.5, edgecolor=edge, facecolor=face,
        )
        ax.add_patch(box)
        ax.text(2.0, y, col, ha="center", va="center",
                fontsize=11, color=text_color, fontweight=weight,
                fontstyle="normal" if is_kept else "italic")
        if not is_kept:
            ax.plot([0.8, 3.2], [y, y], color="#9CA3AF",
                    linewidth=1.2, alpha=0.7)

    # Arrow
    arrow = FancyArrowPatch(
        (3.7, 5.0), (5.9, 5.0),
        arrowstyle="-|>", mutation_scale=25,
        linewidth=2.5, color=ACCENT,
    )
    ax.add_patch(arrow)
    ax.text(4.8, 5.5, "drop", ha="center", fontsize=11,
            color=ACCENT, fontweight="bold")

    # Right panel: kept
    ax.text(8.0, 9.2, "KEPT", fontsize=13, fontweight="bold",
            color=ACCENT, ha="center")
    ax.text(8.0, 8.7, "3 columns", fontsize=11, color=MUTED, ha="center")

    kept_list = ["utterance", "chatTime", "intentClass"]
    descriptions = ["input text", "→ [TIME=x.xx] token", "target label E/I/A/O"]
    for i, (col, desc) in enumerate(zip(kept_list, descriptions)):
        y = 7.2 - i * 1.4
        box = FancyBboxPatch(
            (6.2, y - 0.35), 3.6, 0.7,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=2, edgecolor=ACCENT, facecolor="#DBEAFE",
        )
        ax.add_patch(box)
        ax.text(8.0, y + 0.08, col, ha="center", va="center",
                fontsize=13, color=INK, fontweight="bold")
        ax.text(8.0, y - 0.22, desc, ha="center", va="center",
                fontsize=10, color=MUTED, fontstyle="italic")

    ax.set_title("Column Reduction — Keep Only What the Model Needs",
                 fontsize=16, fontweight="bold", color=INK, pad=10, loc="center")

    plt.tight_layout()
    out = OUT_DIR / "02_columns_reduction.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved {out}")


def chart_before_after():
    """Before/after row showing [SEPA] removal + chatTime normalization."""
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # BEFORE
    ax.text(0.3, 8.7, "BEFORE", fontsize=13, fontweight="bold", color=MUTED)
    before_box = FancyBboxPatch(
        (0.3, 5.6), 9.4, 2.7,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=1.5, edgecolor="#D1D5DB", facecolor="#F9FAFB",
    )
    ax.add_patch(before_box)

    ax.text(0.7, 7.7, "utterance:", fontsize=11, color=MUTED, fontweight="bold")
    ax.text(2.4, 7.7, '"ur trash [SEPA] noob report"',
            fontsize=12, color=INK, family="monospace")
    ax.text(0.7, 6.95, "chatTime:", fontsize=11, color=MUTED, fontweight="bold")
    ax.text(2.4, 6.95, "2987   (raw seconds)",
            fontsize=12, color=INK, family="monospace")
    ax.text(0.7, 6.2, "intentClass:", fontsize=11, color=MUTED, fontweight="bold")
    ax.text(2.4, 6.2, "E", fontsize=12, color=INK,
            family="monospace", fontweight="bold")

    # Arrow
    arrow = FancyArrowPatch(
        (5.0, 5.4), (5.0, 4.4),
        arrowstyle="-|>", mutation_scale=25,
        linewidth=2.5, color=ACCENT,
    )
    ax.add_patch(arrow)
    ax.text(5.3, 4.9, "preprocess", fontsize=11,
            color=ACCENT, fontweight="bold", va="center")

    # AFTER
    ax.text(0.3, 3.8, "AFTER", fontsize=13, fontweight="bold", color=ACCENT)
    after_box = FancyBboxPatch(
        (0.3, 0.7), 9.4, 2.7,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=2, edgecolor=ACCENT, facecolor="#DBEAFE",
    )
    ax.add_patch(after_box)

    ax.text(0.7, 2.8, "utterance:", fontsize=11, color=MUTED, fontweight="bold")
    ax.text(2.4, 2.8, '"ur trash noob report"',
            fontsize=12, color=INK, family="monospace")
    ax.text(0.7, 2.05, "chatTime:", fontsize=11, color=MUTED, fontweight="bold")
    ax.text(2.4, 2.05, "0.834   →   [TIME=0.83]",
            fontsize=12, color=INK, family="monospace", fontweight="bold")
    ax.text(0.7, 1.3, "intentClass:", fontsize=11, color=MUTED, fontweight="bold")
    ax.text(2.4, 1.3, "E", fontsize=12, color=INK,
            family="monospace", fontweight="bold")

    ax.set_title("Sample Transformation — One Row, Before vs. After",
                 fontsize=16, fontweight="bold", color=INK, pad=8, loc="center")

    plt.tight_layout()
    out = OUT_DIR / "03_before_after.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved {out}")


def chart_pipeline():
    """Horizontal pipeline diagram of preprocessing stages."""
    fig, ax = plt.subplots(figsize=(13, 3.8), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    stages = [
        ("Raw\nCSV", "44,869 rows\n10 cols", "#F3F4F6", "#9CA3AF"),
        ("Drop\nColumns", "10 → 3", "#DBEAFE", ACCENT),
        ("Normalize\nchatTime", "[-90, 3600]\n→ [0, 1]", "#DBEAFE", ACCENT),
        ("Strip\n[SEPA]", "merge artifacts", "#DBEAFE", ACCENT),
        ("Drop NaN\n+ Validate", "labels ∈ {E,I,A,O}", "#DBEAFE", ACCENT),
        ("Cleaned\nCSV", "ready for\nDistilBERT", "#D1FAE5", "#10B981"),
    ]

    box_w = 1.85
    gap = 0.30
    total_w = len(stages) * box_w + (len(stages) - 1) * gap
    x_start = (14 - total_w) / 2

    for i, (title, sub, face, edge) in enumerate(stages):
        x = x_start + i * (box_w + gap)
        y = 1.6
        box = FancyBboxPatch(
            (x, y), box_w, 1.8,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            linewidth=2, edgecolor=edge, facecolor=face,
        )
        ax.add_patch(box)
        ax.text(x + box_w / 2, y + 1.25, title,
                ha="center", va="center",
                fontsize=12, fontweight="bold", color=INK)
        ax.text(x + box_w / 2, y + 0.45, sub,
                ha="center", va="center",
                fontsize=9.5, color=MUTED, fontstyle="italic")

        if i < len(stages) - 1:
            ax_x = x + box_w + 0.02
            arrow = FancyArrowPatch(
                (ax_x, y + 0.9), (ax_x + gap - 0.04, y + 0.9),
                arrowstyle="-|>", mutation_scale=18,
                linewidth=2, color=INK,
            )
            ax.add_patch(arrow)

    ax.set_title("Preprocessing Pipeline",
                 fontsize=17, fontweight="bold", color=INK, pad=12)

    plt.tight_layout()
    out = OUT_DIR / "04_pipeline.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved {out}")


def chart_time_normalization():
    """Visual for chatTime → [TIME=x.xx] normalization."""
    import pandas as pd
    df = pd.read_csv(Path(__file__).parent / "data" / "CONDA_train.csv")
    raw = df["chatTime"].values

    floor, ceiling = -90, 3600
    clipped = np.clip(raw, floor, ceiling)
    normalized = np.round((clipped - floor) / (ceiling - floor), 3)

    fig = plt.figure(figsize=(14, 9), dpi=200)
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1],
                          hspace=0.55, wspace=0.22,
                          left=0.07, right=0.97, top=0.97, bottom=0.08)

    # --- TOP LEFT: raw distribution with clip boundaries ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(raw, bins=60, color="#9CA3AF", edgecolor="white", linewidth=0.5)
    ax1.axvline(floor, color="#EF4444", linestyle="--", linewidth=2,
                label=f"floor = {floor}")
    ax1.axvline(ceiling, color="#EF4444", linestyle="--", linewidth=2,
                label=f"ceiling = {ceiling}")
    ax1.set_title("Raw  chatTime  (seconds)",
                  fontsize=13, fontweight="bold", color=INK, pad=8)
    ax1.set_xlabel("seconds", fontsize=10, color=MUTED)
    ax1.set_ylabel("count", fontsize=10, color=MUTED)
    ax1.legend(frameon=False, fontsize=9, loc="upper right")
    ax1.text(0.02, 0.95,
             f"min: {int(raw.min())}\nmax: {int(raw.max())}\nμ:    {raw.mean():.0f}",
             transform=ax1.transAxes, fontsize=9, color=MUTED,
             family="monospace", verticalalignment="top",
             bbox=dict(facecolor="#F9FAFB", edgecolor="#E5E7EB",
                       boxstyle="round,pad=0.4"))

    # --- TOP RIGHT: normalized distribution ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(normalized, bins=60, color=ACCENT, edgecolor="white",
             linewidth=0.5, alpha=0.85)
    ax2.set_title("Normalized  →  [TIME=x.xx]",
                  fontsize=13, fontweight="bold", color=ACCENT, pad=8)
    ax2.set_xlabel("normalized value", fontsize=10, color=MUTED)
    ax2.set_ylabel("count", fontsize=10, color=MUTED)
    ax2.set_xlim(-0.02, 1.02)
    ax2.text(0.02, 0.95,
             f"min: {normalized.min():.2f}\nmax: {normalized.max():.2f}\nμ:    {normalized.mean():.2f}",
             transform=ax2.transAxes, fontsize=9, color=MUTED,
             family="monospace", verticalalignment="top",
             bbox=dict(facecolor="#EFF6FF", edgecolor="#BFDBFE",
                       boxstyle="round,pad=0.4"))

    # --- BOTTOM: formula + mapping examples ---
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_xlim(0, 14)
    ax3.set_ylim(0, 5)
    ax3.axis("off")

    # Formula box
    formula_box = FancyBboxPatch(
        (0.5, 3.0), 13, 1.6,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=2, edgecolor=ACCENT, facecolor="#EFF6FF",
    )
    ax3.add_patch(formula_box)
    ax3.text(7, 4.2, "Normalization Formula",
             ha="center", fontsize=11, color=MUTED, fontweight="bold")
    ax3.text(7, 3.45,
             r"$x_{\mathrm{norm}} = \frac{\mathrm{clip}(x,\ -90,\ 3600) - (-90)}{3600 - (-90)}$",
             ha="center", fontsize=20, color=INK)

    # Example mappings on a number line
    examples = [
        (-492, 0.000, "clipped"),
        (1816, 0.516, "median"),
        (2987, 0.834, ""),
        (5477, 1.000, "clipped"),
    ]

    line_y = 1.5
    line_start, line_end = 2.2, 13.2
    ax3.plot([line_start, line_end], [line_y, line_y],
             color=INK, linewidth=1.5)

    for raw_v, norm_v, note in examples:
        x_pos = line_start + norm_v * (line_end - line_start)
        ax3.plot(x_pos, line_y, "o", markersize=10,
                 color=ACCENT, zorder=4, markeredgecolor="white",
                 markeredgewidth=2)
        # Raw value above
        ax3.text(x_pos, line_y + 0.55, f"{raw_v}",
                 ha="center", fontsize=10, color=MUTED,
                 family="monospace")
        # Normalized below
        ax3.text(x_pos, line_y - 0.45, f"[TIME={norm_v:.2f}]",
                 ha="center", fontsize=10, color=ACCENT,
                 fontweight="bold", family="monospace")
        if note:
            ax3.text(x_pos, line_y - 0.9, note,
                     ha="center", fontsize=8.5, color="#EF4444",
                     fontstyle="italic")

    ax3.text(1.7, line_y + 0.55, "raw:",
             ha="right", fontsize=10, color=MUTED, fontweight="bold")
    ax3.text(1.7, line_y - 0.45, "norm:",
             ha="right", fontsize=10, color=ACCENT, fontweight="bold")

    out = OUT_DIR / "05_time_normalization.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved {out}")


def chart_latency():
    """Speed comparison: full BERT vs DistilBERT."""
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=200)
    fig.patch.set_facecolor(BG)

    models = ["BERT-base\n(110M params)", "DistilBERT\n(66M params)"]
    inference_relative = [100, 60]
    colors_bar = ["#9CA3AF", ACCENT]

    bars = ax.barh(models, inference_relative, color=colors_bar,
                   edgecolor="white", linewidth=2, height=0.55)

    for bar, val in zip(bars, inference_relative):
        ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val}%", va="center", fontsize=15,
                fontweight="bold", color=INK)

    ax.axvline(x=100, color="#9CA3AF", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlim(0, 118)
    ax.set_xlabel("Relative inference time  (BERT = 100%)",
                  fontsize=11, color=MUTED)
    ax.tick_params(axis="y", labelsize=12)
    ax.spines["left"].set_visible(False)

    ax.annotate("~40% faster",
                xy=(60, 1), xytext=(80, 0.5),
                fontsize=13, color=ACCENT, fontweight="bold",
                ha="center", va="center",
                bbox=dict(facecolor="#DBEAFE", edgecolor=ACCENT,
                          boxstyle="round,pad=0.4"),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT,
                                linewidth=1.5))

    ax.text(0.5, -0.22,
            "Live gameplay needs millisecond-scale predictions  —  the smaller model wins.",
            transform=ax.transAxes, ha="center",
            fontsize=11, color=MUTED, fontstyle="italic")

    ax.set_title("Latency  —  DistilBERT vs. BERT-base",
                 fontsize=16, fontweight="bold", color=INK, pad=14)

    plt.tight_layout()
    out = OUT_DIR / "06_latency.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved {out}")


def chart_input_length():
    """Histogram of CONDA utterance token lengths with max_length=64 line."""
    import pandas as pd
    from transformers import DistilBertTokenizerFast
    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    df = pd.read_csv(Path(__file__).parent / "data" / "processed" /
                     "CONDA_train_cleaned.csv")
    lengths = df["utterance"].fillna("").apply(lambda s: len(tok.tokenize(s))).values

    pct_under_64 = (lengths <= 64).mean() * 100
    median = int(np.median(lengths))

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=200)
    fig.patch.set_facecolor(BG)

    ax.hist(lengths, bins=range(0, 70, 1), color=ACCENT,
            edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(64, color="#EF4444", linestyle="--", linewidth=2.5,
               label=f"max_length = 64")
    ax.axvline(median, color="#10B981", linestyle="--", linewidth=2,
               label=f"median = {median} tokens")

    ax.set_xlabel("Tokens per utterance", fontsize=11, color=MUTED)
    ax.set_ylabel("Number of utterances", fontsize=11, color=MUTED)
    ax.set_xlim(0, 70)
    ax.legend(frameon=False, fontsize=11, loc="upper right")

    ax.text(0.98, 0.65,
            f"{pct_under_64:.2f}%\nof utterances\n≤ 64 tokens",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=14, color=INK, fontweight="bold",
            bbox=dict(facecolor="#DBEAFE", edgecolor=ACCENT,
                      boxstyle="round,pad=0.6"))

    ax.set_title("Utterance Length Distribution  —  CONDA Training Set",
                 fontsize=16, fontweight="bold", color=INK, pad=14)

    plt.tight_layout()
    out = OUT_DIR / "07_input_length.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved {out}")


def chart_subword_tokenization():
    """Visual showing WordPiece breakdown of game-chat words."""
    examples = [
        ("noobs",      ["no", "##ob", "##s"]),
        ("gg",         ["g", "##g"]),
        ("afk",        ["af", "##k"]),
        ("pudge",      ["pu", "##dge"]),
        ("rekt",       ["re", "##kt"]),
        ("reportiing", ["report", "##ii", "##ng"]),
    ]

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, len(examples) + 1.2)
    ax.axis("off")

    chunk_colors = ["#DBEAFE", "#BFDBFE", "#93C5FD", "#60A5FA"]
    edge_color = ACCENT

    for i, (word, pieces) in enumerate(examples):
        y = len(examples) - i

        ax.text(0.3, y, word, fontsize=15, fontweight="bold",
                color=INK, family="monospace", va="center")

        arrow = FancyArrowPatch(
            (3.2, y), (4.2, y),
            arrowstyle="-|>", mutation_scale=18,
            linewidth=2, color=MUTED,
        )
        ax.add_patch(arrow)

        x_cursor = 4.5
        for j, piece in enumerate(pieces):
            face = chunk_colors[j % len(chunk_colors)]
            text_w = max(0.85, 0.32 * len(piece) + 0.4)

            box = FancyBboxPatch(
                (x_cursor, y - 0.32), text_w, 0.64,
                boxstyle="round,pad=0.02,rounding_size=0.10",
                linewidth=1.5, edgecolor=edge_color, facecolor=face,
            )
            ax.add_patch(box)
            ax.text(x_cursor + text_w / 2, y, piece,
                    ha="center", va="center", fontsize=13,
                    color=INK, family="monospace", fontweight="bold")
            x_cursor += text_w + 0.18

    ax.text(0.3, len(examples) + 0.7, "input word",
            fontsize=10, color=MUTED, fontweight="bold")
    ax.text(4.5, len(examples) + 0.7, "WordPiece sub-tokens",
            fontsize=10, color=ACCENT, fontweight="bold")

    ax.text(6, 0.3,
            "##  prefix marks a continuation of the previous token  —  "
            "no word is ever \"unknown\"",
            ha="center", fontsize=10.5, color=MUTED, fontstyle="italic")

    ax.set_title("Subword Tokenization  —  Game Chat Survives Vocabulary Limits",
                 fontsize=16, fontweight="bold", color=INK, pad=12)

    plt.tight_layout()
    out = OUT_DIR / "08_subword_tokenization.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    chart_class_distribution()
    chart_columns_reduction()
    chart_before_after()
    chart_pipeline()
    chart_time_normalization()
    chart_latency()
    chart_input_length()
    chart_subword_tokenization()
    print(f"\nAll charts saved to: {OUT_DIR}")
