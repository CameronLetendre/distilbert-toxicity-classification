from model.toxicity_classifier import ToxicityClassifier

test_cases = [
    (0.50, "gg wp"),                      # likely O
    (0.50, "noob report this guy"),       # likely E
    (0.50, "uninstall the game"),         # likely I
    (0.50, "push mid"),                   # likely A
    (0.10, "hi everyone good luck"),      # likely O, early game
    (0.95, "trash team carried"),         # likely E or I, late game
    (0.1, "gg")
]

clf = ToxicityClassifier(r"C:\Users\piedu\OneDrive\Documents\GitHub\Project\src\results\checkpoints\best_model")
for t, msg in test_cases:
    label, conf = clf.predict(t, msg)
    print(f"[TIME={t:.2f}] {msg!r:50} → {label} ({conf:.1%})")