"""One-shot diagnostic: capture ROIs, run OCR, dump everything to debug/."""
import json
from pathlib import Path

import cv2
import pytesseract

from ocr.capture_screen import (
    _OCR_READER,
    capture_frame,
    SCORE_CFG,
    TIME_CFG,
)

GAME = "VALORANT"
GAME_INFO_PATH = Path(__file__).parent / "game_info.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "debug"


def clean_image(img):
    img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    return img


def clean_digits(img):
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    return img


def main():
    OUT_DIR.mkdir(exist_ok=True)
    with open(GAME_INFO_PATH) as f:
        rois = json.load(f)[GAME]

    print(f"ROI config: {rois}")
    frames = capture_frame(rois)

    for name, frame in frames.items():
        raw_path = OUT_DIR / f"{name}_raw.png"
        cv2.imwrite(str(raw_path), frame)
        print(f"  wrote {raw_path}  shape={frame.shape}")

    chat_clean = clean_image(frames["chat"])
    time_clean = clean_digits(clean_image(frames["time"]))
    won_clean = clean_digits(clean_image(frames["won"]))
    lose_clean = clean_digits(clean_image(frames["lost"]))

    cv2.imwrite(str(OUT_DIR / "chat_clean.png"), chat_clean)
    cv2.imwrite(str(OUT_DIR / "time_clean.png"), time_clean)
    cv2.imwrite(str(OUT_DIR / "won_clean.png"), won_clean)
    cv2.imwrite(str(OUT_DIR / "lose_clean.png"), lose_clean)

    time_str = pytesseract.image_to_string(time_clean, config=TIME_CFG).strip()
    won_str = pytesseract.image_to_string(won_clean, config=SCORE_CFG).strip()
    lose_str = pytesseract.image_to_string(lose_clean, config=SCORE_CFG).strip()

    print("\n--- Tesseract ---")
    print(f"  time_str = {time_str!r}")
    print(f"  won_str  = {won_str!r}")
    print(f"  lose_str = {lose_str!r}")

    print("\n--- EasyOCR (chat) ---")
    chat_results = _OCR_READER.readtext(chat_clean, detail=1)
    if not chat_results:
        print("  (no text detected)")
    for bbox, text, conf in chat_results:
        print(f"  conf={conf:.2f}  text={text!r}")

    print(f"\nDone. Open the PNGs in {OUT_DIR} to verify the ROIs are aimed correctly.")


if __name__ == "__main__":
    main()
