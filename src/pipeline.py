import json
import os
import time
from pathlib import Path
from time import perf_counter

import cv2

from ocr.capture_screen import extract_chat, extract_match_state, capture_frame
from model.toxicity_classifier import ToxicityClassifier
from visualization.overlay import ScreenOverlay

PROFILE = os.environ.get("PIPELINE_PROFILE", "0") == "1"

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = _PROJECT_ROOT / "src/model/results/checkpoints/best_model"
GAME_INFO_PATH = Path(__file__).parent / "game_info.json"

CAPTURE_INTERVAL_S = 0.4
CHAT_DIFF_THRESHOLD = 0.5         # mean per-pixel absdiff below this = unchanged
NOT_DETECTED_AFTER_N = 15         # consecutive blank ticks before declaring "game not detected"
CLASSIFIED_CACHE_MAX = 512
META_REREAD_S = 8.0               # re-read time/score at most this often (success path)
META_FAIL_COOLDOWN_S = 2.0        # min wait before retrying after a parse failure


def load_game_config(game: str) -> dict:
    with open(GAME_INFO_PATH) as f:
        return json.load(f)[game]


def chat_unchanged(prev, curr, threshold=CHAT_DIFF_THRESHOLD):
    if prev is None or curr is None or prev.shape != curr.shape:
        return False
    return float(cv2.absdiff(prev, curr).mean()) < threshold


def main():
    clf = ToxicityClassifier(str(MODEL_PATH))
    bboxes = load_game_config("VALORANT")
    chat_roi = bboxes["chat"]
    overlay = ScreenOverlay()

    print(f"Starting live capture loop (interval={CAPTURE_INTERVAL_S}s). Ctrl+C to stop.")

    last_chat_img = None
    last_good_time = None
    next_meta_read_at = 0.0    # perf_counter deadline; retry whenever now >= this
    classified_cache = {}      # (text, time_bin) -> (label, conf)
    blank_streak = 0
    announced_not_detected = False

    try:
        while True:
            tick_start = perf_counter()
            t_cap0 = perf_counter()
            roi_data = capture_frame(bboxes)
            t_cap = perf_counter() - t_cap0
            chat_img = roi_data["chat"]

            if chat_unchanged(last_chat_img, chat_img):
                time.sleep(CAPTURE_INTERVAL_S)
                continue

            t_chat0 = perf_counter()
            messages = extract_chat(roi_data)
            t_chat = perf_counter() - t_chat0

            # Refresh match state only periodically. TIME token is normalized over
            # a ~30-min match, so a few seconds of drift is below input granularity.
            # On parse failure, back off for META_FAIL_COOLDOWN_S so we don't retry
            # every tick while scoreboard is mid-animation or off-screen.
            now = perf_counter()
            should_read_meta = bool(messages) and now >= next_meta_read_at
            t_meta = 0.0
            t = None
            if should_read_meta:
                t_meta0 = perf_counter()
                t = extract_match_state(roi_data)
                t_meta = perf_counter() - t_meta0
                if t is not None:
                    last_good_time = t
                    next_meta_read_at = now + META_REREAD_S
                else:
                    next_meta_read_at = now + META_FAIL_COOLDOWN_S

            effective_time = t if t is not None else last_good_time

            # "Game not detected" only when we've had no time anchor AND no messages
            # for several consecutive ticks. Empty chat alone isn't enough — players
            # may simply not be talking.
            if effective_time is None and not messages:
                blank_streak += 1
                if blank_streak >= NOT_DETECTED_AFTER_N:
                    if not announced_not_detected:
                        print("Game not detected - waiting...")
                        announced_not_detected = True
                    overlay.clear()
                    last_chat_img = None
                time.sleep(CAPTURE_INTERVAL_S)
                continue

            blank_streak = 0
            if announced_not_detected:
                print("Game detected - resuming.")
                announced_not_detected = False

            last_chat_img = chat_img

            if not messages:
                overlay.clear()
                time.sleep(CAPTURE_INTERVAL_S)
                continue

            # No timer parse yet (startup): neutral mid-match anchor.
            if effective_time is None:
                effective_time = 0.5
            time_bin = round(effective_time, 3)
            results = [None] * len(messages)
            todo_idx = []
            todo_items = []
            for i, (msg, bbox) in enumerate(messages):
                cached = classified_cache.get((msg, time_bin))
                if cached is not None:
                    results[i] = ((effective_time, msg, bbox), cached)
                else:
                    todo_idx.append(i)
                    todo_items.append((effective_time, msg))

            t_clf0 = perf_counter()
            if todo_items:
                preds = clf.predict_batch(todo_items)
                for i, pred in zip(todo_idx, preds):
                    msg, bbox = messages[i]
                    classified_cache[(msg, time_bin)] = pred
                    results[i] = ((effective_time, msg, bbox), pred)
            t_clf = perf_counter() - t_clf0

            if len(classified_cache) > CLASSIFIED_CACHE_MAX:
                for k in list(classified_cache.keys())[: CLASSIFIED_CACHE_MAX // 2]:
                    del classified_cache[k]

            for (t_, msg, _), (label, conf) in results:
                marker = "" if t is not None else " (time-cached)"
                print(f"[TIME={t_:.2f}]{marker} {msg!r:50} {label} ({conf:.1%})")

            t_ov0 = perf_counter()
            overlay.update(results, chat_roi["left"], chat_roi["top"])
            t_ov = perf_counter() - t_ov0

            if PROFILE:
                total = perf_counter() - tick_start
                meta_str = f"meta={t_meta*1000:.0f}ms" if should_read_meta else "meta=skip"
                print(
                    f"[tick] total={total*1000:.0f}ms "
                    f"cap={t_cap*1000:.0f}ms chat={t_chat*1000:.0f}ms "
                    f"{meta_str} "
                    f"clf={t_clf*1000:.0f}ms (new={len(todo_items)}/{len(messages)}) "
                    f"overlay={t_ov*1000:.0f}ms"
                )

            time.sleep(CAPTURE_INTERVAL_S)
    finally:
        overlay.close()


if __name__ == "__main__":
    main()
