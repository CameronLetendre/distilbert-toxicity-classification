import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax


class ToxicityClassifier:
    LABEL_MAP = {0: "A", 1: "E", 2: "I", 3: "O"}
    MAX_LENGTH = 64

    def __init__(self, model_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device).eval()

    @staticmethod
    def format_input(normalized_time, message):
        return f"[TIME={normalized_time:.2f}] {message.strip()}"
    

    @torch.no_grad()
    def predict(self, normalized_time, message):
        text = self.format_input(normalized_time, message)
        enc = self.tokenizer(
            text,
            max_length=self.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**enc).logits
        probs = softmax(logits, dim=-1)[0]
        idx = probs.argmax().item()
        return self.LABEL_MAP[idx], probs[idx].item()
    
    @torch.no_grad()
    def predict_batch(self, items):
        texts = [self.format_input(t, m) for t, m in items]
        enc = self.tokenizer(
            texts,
            max_length=self.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**enc).logits
        probs = softmax(logits, dim=-1)
        idxs = probs.argmax(dim=-1).cpu().tolist()
        confs = probs.max(dim=-1).values.cpu().tolist()
        return [(self.LABEL_MAP[i], c) for i, c in zip(idxs, confs)]