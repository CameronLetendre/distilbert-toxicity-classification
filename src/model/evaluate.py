from transformers import DistilBertForSequenceClassification
from pathlib import Path
from torch.utils.data import DataLoader
import sys
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# Directories
REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(Path(__file__).parent))
from CONDA_dataset import CONDADataset

checkpoint_dir = REPO_ROOT / "src/model/results/checkpoints/best_model"
TEST_CSV = REPO_ROOT / "model_tuning/data/processed/CONDA_valid_cleaned.csv"
METRICS_OUT = REPO_ROOT / "src/model/results/metrics_with_time.json"

# Label Constants
LABELS = ["A", "E", "I", "O"]            # integer-id order (0,1,2,3)
EVAL_ORDER = ["E", "I", "A", "O"]        # CONDA paper reporting order
EVAL_IDX = [LABELS.index(c) for c in EVAL_ORDER]
RUN_NAME = "with_time"


def compute_accuracy(y_true, y_pred):
    uca = accuracy_score(y_true, y_pred)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3])
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2, 3])
    cm = confusion_matrix(y_true, y_pred, labels=EVAL_IDX)

    print(f"{"UCA:":>15} {uca:.4f}")
    print(f"{"Macro F1:":>15} {macro_f1:.4f}")

    for name in EVAL_ORDER:
        print(f"{name:>10}: {per_class_f1[LABELS.index(name)]:.4f}")

    print(f"\nConfusion Matrix (rows=true, cols=pred; order={EVAL_ORDER}):")
    print(cm)

    metrics = {
        "run": RUN_NAME,
        "uca": float(uca),
        "macro_f1": float(macro_f1),
        "per_class_f1": {n: float(per_class_f1[LABELS.index(n)]) for n in EVAL_ORDER},
        "confusion_matrix": cm.tolist(),
        "label_order": EVAL_ORDER,
    }
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics -> {METRICS_OUT}")


# Load model and data and set to eval
model = DistilBertForSequenceClassification.from_pretrained(str(checkpoint_dir))

test_dataset = CONDADataset(csv_path=TEST_CSV)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

y_true, y_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits
        y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
        y_true.extend(batch["label"].numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

compute_accuracy(y_true, y_pred)