"""Render the 'Why DistilBERT' slide graphic as a PNG for PowerPoint."""
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = r"c:\Users\piedu\Documents\Fall 2025\Numerical\Project\Project\presentations\why_distilbert.png"

BG = "#0f172a"
PANEL = "#1e293b"
EDGE = "#475569"
CYAN = "#38bdf8"
INDIGO = "#818cf8"
TEXT = "#f1f5f9"
MUTED = "#94a3b8"
DIM = "#334155"
GREEN = "#22c55e"

fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
ax.set_xlim(0, 1280)
ax.set_ylim(0, 720)
ax.invert_yaxis()
ax.set_axis_off()
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

def panel(x, y, w, h, edge=EDGE, fill=PANEL, lw=1.5, radius=12):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         linewidth=lw, edgecolor=edge, facecolor=fill)
    ax.add_patch(box)

def text(x, y, s, **kw):
    kw.setdefault("color", TEXT)
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    ax.text(x, y, s, **kw)

def arrow(x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=18,
                        color=MUTED, linewidth=2)
    ax.add_patch(a)

# Title
text(640, 50, "Why DistilBERT Fits Live Game Chat",
     fontsize=22, fontweight="bold")
ax.plot([440, 840], [78, 78], color=CYAN, linewidth=3)

# Stage 1: raw chat
panel(60, 140, 260, 90)
text(190, 165, "RAW CHAT", color=MUTED, fontsize=9, fontweight="bold")
text(190, 192, '"ur trash noob report"', fontsize=12, fontstyle="italic")
text(190, 215, "[TIME=0.84]", color="#64748b", fontsize=9)

arrow(330, 185, 385, 185)

# Stage 2: subword tokens
panel(395, 135, 380, 100, edge=CYAN)
text(585, 158, "SUBWORD TOKENIZATION", color=CYAN, fontsize=9, fontweight="bold")

tokens = [("ur", 44, CYAN), ("trash", 62, CYAN), ("no", 44, CYAN),
          ("##ob", 58, INDIGO), ("re", 44, CYAN), ("##port", 58, INDIGO)]
tx = 413
for tok, w, color in tokens:
    panel(tx, 183, w, 32, edge=color, fill="#0f172a", lw=1.2, radius=6)
    text(tx + w/2, 199, tok, color="#e0f2fe", fontsize=10, fontweight="bold")
    tx += w + 8

arrow(785, 185, 840, 185)

# Stage 3: 64-slot grid
panel(850, 130, 370, 110)
text(1035, 158, "MAX LENGTH = 64 TOKENS", color=MUTED, fontsize=9, fontweight="bold")

# 8x8 grid
gx0, gy0 = 885, 175
size, gap = 22, 4
filled_colors = [CYAN, CYAN, CYAN, INDIGO, CYAN, INDIGO]
for i in range(16):
    row, col = i // 8, i % 8
    x = gx0 + col * (size + gap)
    y = gy0 + row * (size + gap)
    if i < 6:
        c = filled_colors[i]
        panel(x, y, size, size, edge=c, fill=c, lw=0, radius=3)
    else:
        panel(x, y, size, size, edge=DIM, fill=DIM, lw=0, radius=3)

text(1140, 192, "6 used", color="#cbd5e1", fontsize=10, ha="center")
text(1140, 210, "58 padded", color="#64748b", fontsize=10, ha="center")
text(1140, 228, "→ short inputs", color=GREEN, fontsize=9, fontweight="bold", ha="center")

arrow(640, 260, 640, 320)

# DistilBERT block (gradient simulated with two stacked rectangles)
panel(420, 330, 440, 90, edge=INDIGO, fill=INDIGO, lw=0, radius=14)
panel(420, 330, 220, 90, edge=CYAN, fill=CYAN, lw=0, radius=14)
# Cover the seam to blend
panel(420, 330, 440, 90, edge="#5b8def", fill="none", lw=0, radius=14)
text(640, 358, "DistilBERT-CONDA", color="#0f172a", fontsize=16, fontweight="bold")
text(640, 384, "6 layers · 768 hidden · ~2× faster than BERT",
     color="#0f172a", fontsize=10, fontweight="medium")
text(640, 404, "[CLS] → classification head", color="#1e293b", fontsize=9)

arrow(640, 430, 640, 465)

# Output labels
labels = [("E", "Explicit", "#dc2626"),
          ("I", "Implicit", "#f59e0b"),
          ("A", "Action", "#3b82f6"),
          ("O", "Other", "#10b981")]
lx = 450
for code, name, color in labels:
    panel(lx, 475, 70, 36, edge=color, fill=color, lw=0, radius=8)
    text(lx + 35, 493, code, color="white", fontsize=12, fontweight="bold")
    text(lx + 35, 525, name, color=MUTED, fontsize=9)
    lx += 90

# Bottom takeaways
def badge(cx, cy, ring_color, glyph, glyph_color, glyph_size,
          title, line1, line2):
    circ = mp.Circle((cx, cy), 26, facecolor="#0f172a",
                     edgecolor=ring_color, linewidth=2)
    ax.add_patch(circ)
    text(cx, cy + 2, glyph, color=glyph_color, fontsize=glyph_size,
         fontweight="bold")
    text(cx + 60, cy - 12, title, color=TEXT, fontsize=11,
         fontweight="bold", ha="left")
    text(cx + 60, cy + 6, line1, color=MUTED, fontsize=9, ha="left")
    text(cx + 60, cy + 22, line2, color=MUTED, fontsize=9, ha="left")

badge(90, 630, CYAN, "⚡", CYAN, 18,
      "Low Latency", "Real-time on live", "gameplay")
badge(510, 630, INDIGO, "64", INDIGO, 14,
      "Short Inputs", "Max length 64 — no",
      "need for larger models")
badge(910, 630, GREEN, "≡", GREEN, 18,
      "Subword Tokens", "Handles game slang",
      "& typos gracefully")

plt.savefig(OUT, dpi=200, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
print(f"Wrote {OUT}")
