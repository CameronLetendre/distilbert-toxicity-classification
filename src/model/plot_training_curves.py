"""Parse train logs and produce training_curves.png (loss + macro-F1)."""
import re
from pathlib import Path
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS = {
    "no time token":   REPO_ROOT / "logs/train_no_time_seed42.log",
    "with [TIME]":     REPO_ROOT / "logs/train_with_time_seed42.log",
}
OUT = REPO_ROOT / "src/model/results/plots/training_curves.png"


def parse(log_path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    epochs, train, val, f1 = [], [], [], []
    blocks = re.split(r"^Epoch (\d+)/\d+\s*$", text, flags=re.MULTILINE)
    for i in range(1, len(blocks), 2):
        ep = int(blocks[i])
        body = blocks[i + 1]
        m_tr = re.search(r"^Train Loss:\s+([0-9.]+)", body, re.MULTILINE)
        m_va = re.search(r"^Val Loss:\s+([0-9.]+)",   body, re.MULTILINE)
        m_f1 = re.search(r"^Macro F1:\s+([0-9.]+)",    body, re.MULTILINE)
        if not (m_tr and m_va and m_f1):
            continue
        # Last epoch may appear twice due to header preamble; keep last occurrence
        if epochs and epochs[-1] == ep:
            train[-1], val[-1], f1[-1] = float(m_tr.group(1)), float(m_va.group(1)), float(m_f1.group(1))
        else:
            epochs.append(ep)
            train.append(float(m_tr.group(1)))
            val.append(float(m_va.group(1)))
            f1.append(float(m_f1.group(1)))
    return epochs, train, val, f1


def main():
    runs = {name: parse(p) for name, p in LOGS.items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(11, 4.2))

    colors = {"no time token": "#94a3b8", "with [TIME]": "#38bdf8"}
    for name, (ep, tr, va, f1) in runs.items():
        c = colors[name]
        ax_loss.plot(ep, tr, "-o", color=c, label=f"{name} (train)")
        ax_loss.plot(ep, va, "--s", color=c, label=f"{name} (val)")
        ax_f1.plot(ep, f1, "-o", color=c, label=name)

    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Train / validation loss")
    ax_loss.grid(linestyle=":", alpha=0.5); ax_loss.legend(fontsize=9)

    ax_f1.set_xlabel("Epoch"); ax_f1.set_ylabel("Validation macro-F1")
    ax_f1.set_title("Validation macro-F1")
    ax_f1.grid(linestyle=":", alpha=0.5); ax_f1.legend(fontsize=9)

    for ax in (ax_loss, ax_f1):
        ax.set_xticks(sorted({e for ep, _, _, _ in runs.values() for e in ep}))

    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print(f"wrote {OUT}")
    for name, (ep, tr, va, f1) in runs.items():
        print(f"{name}: epochs={ep} train={tr} val={va} macroF1={f1}")


if __name__ == "__main__":
    main()
