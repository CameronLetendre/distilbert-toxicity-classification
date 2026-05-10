"""Pie chart of CONDA training-split intent class imbalance."""
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent / "class_imbalance_pie.png"

# Counts from the CONDA training split (CLAUDE.md)
counts = {"O": 19982, "E": 3528, "A": 1719, "I": 1692}
names = {
    "E": "Explicit toxicity",
    "I": "Implicit toxicity",
    "A": "Action",
    "O": "Other / neutral",
}
colors = {
    "E": "#dc2626",
    "I": "#f59e0b",
    "A": "#3b82f6",
    "O": "#10b981",
}

labels = list(counts.keys())
sizes = [counts[k] for k in labels]
total = sum(sizes)
explode = [0.0 if k == "O" else 0.04 for k in labels]
slice_colors = [colors[k] for k in labels]

fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=[f"{k} — {names[k]}" for k in labels],
    colors=slice_colors,
    autopct=lambda p: f"{p:.1f}%\n({int(round(p*total/100)):,})",
    startangle=90,
    explode=explode,
    pctdistance=0.72,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    textprops={"fontsize": 11},
)
for t in autotexts:
    t.set_color("white")
    t.set_fontsize(10)
    t.set_fontweight("bold")

ax.set_title(
    f"CONDA training split — intent class imbalance\n"
    f"n = {total:,} utterances",
    fontsize=13, fontweight="bold", pad=18,
)
ax.axis("equal")

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
