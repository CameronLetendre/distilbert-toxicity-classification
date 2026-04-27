from model.toxicity_classifier import ToxicityClassifier

test_cases = [
    (0.50, "gg"),                      # likely O
    (0.50, "ggwp"),       # likely E
    (0.50, "you will never be good at this game"),         # likely I
    (0.50, "report viper"),                   # likely A
    (0.10, "gg"),      # likely O, early game
]
clf = ToxicityClassifier(r"C:\Users\piedu\OneDrive\Documents\GitHub\Project\src\model\results\checkpoints\best_model")
for t, msg in test_cases:
    label, conf = clf.predict(t, msg)
    print(f"[TIME={t:.2f}] {msg!r:50} → {label} ({conf:.1%})")