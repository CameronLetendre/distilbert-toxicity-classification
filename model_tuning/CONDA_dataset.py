import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer
import pandas as pd


class CONDADataset(Dataset):
    """
    PyTorch Dataset for CONDA Dota 2 chat intent classification.
    
    Combines utterance text with a prepended game time token [TIME=x.xx]
    and tokenizes for DistilBERT (cased) input.
    
    Args:
        csv_path: Path to preprocessed CSV with columns: utterance, chatTime, intentClass
        max_length: Maximum token sequence length (default 64)
    """

    LABEL_MAP = {"A": 0, "E": 1, "I": 2, "O": 3}

    def __init__(self, csv_path, max_length=64):
        self.df = pd.read_csv(csv_path)
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Format the time token and combine with utterance
        time_token = f"[TIME={row['chatTime']:.2f}]"
        text = f"{time_token} {row['utterance']}"

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Encode label
        label = self.LABEL_MAP[row["intentClass"]]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


def main():
    dataset = CONDADataset("data/processed/CONDA_train_cleaned.csv")

    print(f"Dataset size: {len(dataset)}")
    print(f"Label mapping: {CONDADataset.LABEL_MAP}")

    sample = dataset[0]
    print(f"\nSample input_ids shape: {sample['input_ids'].shape}")
    print(f"Sample attention_mask shape: {sample['attention_mask'].shape}")
    print(f"Sample label: {sample['label']}")

    decoded = dataset.tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
    print(f"Decoded text: {decoded}")

    lengths = []
    for i in range(min(500, len(dataset))):
        mask = dataset[i]["attention_mask"]
        lengths.append(mask.sum().item())

    print(f"\nToken length stats (first 500 samples):")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Avg: {sum(lengths) / len(lengths):.1f}")


if __name__ == "__main__":
    main()