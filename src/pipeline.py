import json
import time
from pathlib import Path

from ocr.capture_screen import extract_OCR, capture_frame
from model.toxicity_classifier import ToxicityClassifier

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = _PROJECT_ROOT / "src/results/checkpoints/best_model"
GAME_INFO_PATH = Path(__file__).parent / "game_info.json"
TEST_IMAGES_DIR = _PROJECT_ROOT / "screen_capture" / "test_images"

CAPTURE_INTERVAL_S = 2.0
HEARTBEAT_EVERY_N_EMPTY = 15


def load_game_config(game: str) -> dict:
    with open(GAME_INFO_PATH) as f:
        return json.load(f)[game]


def run_pipeline_once(roi_data, classifier):
    timed_messages = extract_OCR(roi_data)
    if not timed_messages:
        return []
    predictions = classifier.predict_batch(timed_messages)
    return list(zip(timed_messages, predictions))


def main():
    clf = ToxicityClassifier(str(MODEL_PATH))
    bboxes = load_game_config("VALORANT")

    print(f"Starting live capture loop (interval={CAPTURE_INTERVAL_S}s). Ctrl+C to stop.")
    empty_streak = 0
    while True:
        roi_data = capture_frame(bboxes)
        results = run_pipeline_once(roi_data, clf)

        if results:
            if empty_streak >= HEARTBEAT_EVERY_N_EMPTY:
                print("Game detected - resuming.")
            empty_streak = 0
            for (t, msg), (label, conf) in results:
                print(f"[TIME={t:.2f}] {msg!r:50} {label} ({conf:.1%})")
        else:
            empty_streak += 1
            if empty_streak % HEARTBEAT_EVERY_N_EMPTY == 0:
                seconds = empty_streak * CAPTURE_INTERVAL_S
                print(f"Game not detected - waiting... ({seconds:.0f}s)")

        time.sleep(CAPTURE_INTERVAL_S)



if __name__ == "__main__":
    main()