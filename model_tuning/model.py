
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import numpy as np
import os
from CONDA_dataset import CONDADataset


# --- Configuration ---
CONFIG = {
    "model_name": "distilbert-base-cased",
    "num_labels": 4,
    "max_length": 64,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "epochs": 5,
    "warmup_ratio": 0.1,
    "output_dir": "results/checkpoints",
    "data_dir": "data/processed",
}

LABEL_MAP = {"A": 0, "E": 1, "I": 2, "O": 3}
LABEL_NAMES = ["A", "E", "I", "O"]

# Class weights: computed as total / (num_classes × class_count) from training split
CLASS_WEIGHTS = torch.tensor([3.915, 1.907, 3.977, 0.337])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE.type)


def create_dataloaders(config):
    """Load train and validation datasets into DataLoaders."""
    train_dataset = CONDADataset(
        csv_path=os.path.join(config["data_dir"], "CONDA_train_cleaned.csv"),
        max_length=config["max_length"]
    )
    val_dataset = CONDADataset(
        csv_path=os.path.join(config["data_dir"], "CONDA_valid_cleaned.csv"),
        max_length=config["max_length"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader


def train_one_epoch(model, train_loader, optimizer, scheduler, criterion):
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion):
    """Run validation. Returns loss, macro F1, per-class F1, and full report."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    per_class_f1 = f1_score(all_labels, all_preds, average=None)

    report = classification_report(
        all_labels, all_preds,
        target_names=LABEL_NAMES,
        digits=4
    )

    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, macro_f1, per_class_f1, report, cm


def train(config):
    """Full training loop with validation and best model saving."""
    print(f"Using device: {DEVICE}")
    print(f"Configuration: {config}\n")

    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)

    # Load data
    train_loader, val_loader = create_dataloaders(config)

    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_labels"]
    ).to(DEVICE)

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE))

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    # Learning rate scheduler with linear warmup
    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    best_macro_f1 = 0.0

    for epoch in range(config["epochs"]):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print(f"{'='*60}")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss, macro_f1, per_class_f1, report, cm = evaluate(model, val_loader, criterion)
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Macro F1:   {macro_f1:.4f}")
        print(f"\nPer-class F1: ", end="")
        for name, score in zip(LABEL_NAMES, per_class_f1):
            print(f"{name}={score:.4f}  ", end="")
        print()

        print(f"\n{report}")
        print(f"Confusion Matrix:\n{cm}")

        # Save best model based on macro F1
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            save_path = os.path.join(config["output_dir"], "best_model")
            model.save_pretrained(save_path)
            train_loader.dataset.tokenizer.save_pretrained(save_path)

            train_loader.dataset.tokenizer.save_pretrained(save_path)
            print(f"\nNew best model saved! Macro F1: {macro_f1:.4f}")

    print(f"\n{'='*60}")
    print(f"Training complete. Best Macro F1: {best_macro_f1:.4f}")
    print(f"Best model saved to: {os.path.join(config['output_dir'], 'best_model')}")
    print(f"{'='*60}")

    return model


if __name__ == "__main__":
    model = train(CONFIG)