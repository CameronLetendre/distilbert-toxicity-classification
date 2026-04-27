import os
import re
import mss
import mss.tools
from time import sleep, perf_counter
import easyocr
import cv2
import pytesseract
import numpy as np
import torch


_TESSERACT_DEFAULT = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(_TESSERACT_DEFAULT):
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_DEFAULT

_USE_GPU = torch.cuda.is_available()
_OCR_READER = easyocr.Reader(['en'], gpu=_USE_GPU)
print(f"[capture_screen] EasyOCR device={_OCR_READER.device} (torch.cuda.is_available={_USE_GPU})")

PROFILE = os.environ.get("PIPELINE_PROFILE", "0") == "1"

TIME_CFG = '--psm 7 -c tessedit_char_whitelist=0123456789:'
SCORE_CFG = '--psm 7 -c tessedit_char_whitelist=0123456789'

_TIME_RE = re.compile(r'(\d{1,2}):?(\d{2})')
_SCORE_RE = re.compile(r'\d+')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def capture_frame(game):
    with mss.mss() as sct:
        return {
            roi: cv2.cvtColor(np.array(sct.grab(game[roi])), cv2.COLOR_BGRA2BGR)
            for roi in game
        }


def _clean_image(img):
    if isinstance(img, str):
        img = cv2.imread(img)
    img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    return img


def _clean_digits(img):
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    return img


def _union(boxes):
    xs = [pt[0] for b in boxes for pt in b]
    ys = [pt[1] for b in boxes for pt in b]
    return (min(xs), min(ys), max(xs), max(ys))


def _normalize_time(elapsed_seconds, floor=-90, ceiling=3600):
    clipped = max(floor, min(3600, elapsed_seconds))
    return (clipped - -90) / (ceiling - -90)


def _extract_chatbox(chat_text):
    messages = []
    curr = []
    curr_bbox = []
    for bbox, text, conf in chat_text:
        if conf < 0.3:
            continue
        if ":" in text:
            if curr:
                messages.append((" ".join(curr), _union(curr_bbox)))
            text = text[text.find(":") + 2:]
            curr = [text]
            curr_bbox = [bbox]
        else:
            if curr:
                curr.append(text)
                curr_bbox.append(bbox)

    if curr:
        messages.append((" ".join(curr), _union(curr_bbox)))

    return messages


def _parse_time(time_str, won_str, lose_str):
    m = _TIME_RE.search(time_str)
    if not m:
        return None
    mins, secs = int(m.group(1)), int(m.group(2))
    if secs >= 100:
        return None
    round_seconds = mins * 60 + secs

    won = _SCORE_RE.search(won_str)
    lose = _SCORE_RE.search(lose_str)
    if not won or not lose:
        return None
    completed_rounds = (int(won.group(0)) + int(lose.group(0))) - 1
    elapsed_in_current_round = 100 - round_seconds
    total_elapsed = (completed_rounds * 100) + elapsed_in_current_round

    return _normalize_time(total_elapsed)


def extract_chat(information):
    """Run EasyOCR on the chat ROI only. Returns list of (text, bbox)."""
    t0 = perf_counter()
    chat_img = _clean_image(information["chat"])
    t_clean = perf_counter() - t0

    t0 = perf_counter()
    raw = _extract_chatbox(_OCR_READER.readtext(chat_img, detail=1))
    t_easyocr = perf_counter() - t0
    messages = [(msg.strip(), tuple(c / 4 for c in bbox)) for msg, bbox in raw]

    if PROFILE:
        h, w = chat_img.shape[:2]
        print(f"[ocr-chat] chat(upscaled)={w}x{h} clean={t_clean*1000:.0f}ms easyocr={t_easyocr*1000:.0f}ms msgs={len(messages)}")

    return messages


def _read_digits(img, allowlist):
    """EasyOCR on a small digit ROI. Returns space-joined recognized text."""
    results = _OCR_READER.readtext(img, allowlist=allowlist, detail=0, paragraph=False)
    return " ".join(results) if results else ""


def extract_match_state(information):
    """Run digit OCR on time + score ROIs via EasyOCR (already loaded, GPU)."""
    t0 = perf_counter()
    time_img = _clean_image(information["time"])
    win_img  = _clean_image(information["won"])
    lose_img = _clean_image(information["lost"])
    t_clean = perf_counter() - t0

    t0 = perf_counter()
    time_str = _read_digits(_clean_digits(time_img), allowlist='0123456789:')
    won_str  = _read_digits(_clean_digits(win_img),  allowlist='0123456789')
    lose_str = _read_digits(_clean_digits(lose_img), allowlist='0123456789')
    t_ocr = perf_counter() - t0
    t = _parse_time(time_str, won_str, lose_str)

    if t is None:
        debug_dir = os.path.join(_SCRIPT_DIR, "..", "..", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "time_fail_clean.png"), _clean_digits(time_img))
        cv2.imwrite(os.path.join(debug_dir, "time_fail_raw.png"), information["time"])

    if PROFILE:
        time_repr = '%.3f' % t if t is not None else f"None (raw: t={time_str!r} w={won_str!r} l={lose_str!r})"
        print(f"[ocr-meta] clean={t_clean*1000:.0f}ms easyocr={t_ocr*1000:.0f}ms time={time_repr}")

    return t


def extract_OCR(information):
    """Backwards-compatible: runs both stages."""
    messages = extract_chat(information)
    t = extract_match_state(information)
    return t, messages
