import os
import mss
import mss.tools
from time import sleep, perf_counter
import easyocr
import cv2
import pytesseract
import numpy as np



_TESSERACT_DEFAULT = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(_TESSERACT_DEFAULT):
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_DEFAULT

_OCR_READER = easyocr.Reader(['en'])

TIME_CFG = '--psm 6 -c tessedit_char_whitelist=0123456789:'
SCORE_CFG = '--psm 6 -c tessedit_char_whitelist=0123456789'

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def capture_frame(game):
    with mss.mss() as sct:
        return {
            roi: cv2.cvtColor(np.array(sct.grab(game[roi])), cv2.COLOR_BGRA2BGR)
            for roi in game
        }


def extract_OCR(information):
    def clean_image(img):
        if isinstance(img, str):
            img = cv2.imread(img)
        img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.bitwise_not(img)
        return img

    def clean_digits(img):
        _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
        return img
    
    def union(boxes):
        xs = [pt[0] for b in boxes for pt in b]
        ys = [pt[1] for b in boxes for pt in b]

        return (min(xs), min(ys), max(xs), max(ys))
    
    def normalize_time(elapsed_seconds, floor=-90, ceiling=3600):
        clipped = max(floor, min(3600, elapsed_seconds))
        return (clipped - -90) / (ceiling - -90)
    
    def extract_chatbox(chat_text):
        messages = []
        curr = []
        curr_bbox = []
        for bbox, text, conf in chat_text:
            if ":" in text:
                if curr:
                    messages.append(" ".join(curr))
                text = text[text.find(":")+2:]
                curr = [text]
                curr_bbox = [bbox]

            else:
                if curr:                 
                    curr.append(text)
                    curr_bbox.append(bbox)

        if curr:
            messages.append(" ".join(curr))

        return messages


    def extract_time(times_text):
        times_text = [x[0] if x else "" for x in times_text]
        times_text[0] = times_text[0][:-2] + ":" + times_text[0][-2:]

        raw = times_text[0].replace(":", "")
        if len(raw) == 3:
            raw = f"{raw[0]}:{raw[1:]}"
        elif len(raw) == 4:
            raw = f"{raw[:2]}:{raw[2:]}"
        else:
            return None

        mins, secs = raw.split(":")
        total_elapsed = (int(mins) * 60) + int(secs)

        if len(times_text) > 1:
            if not (times_text[1].isdigit() and times_text[2].isdigit()):
                return None
            completed_rounds = (int(times_text[1]) + int(times_text[2])) - 1
            elapsed_in_current_round = 100 - total_elapsed
            total_elapsed = (completed_rounds * 100) + elapsed_in_current_round

        return normalize_time(total_elapsed)


    chat_img = clean_image(information["chat"])
    time_img = clean_image(information["time"])
    win_img  = clean_image(information["won"])
    lose_img = clean_image(information["lost"])

    messages = extract_chatbox(_OCR_READER.readtext(chat_img, detail = 1))

    time_str = pytesseract.image_to_string(clean_digits(time_img), config=TIME_CFG).strip()
    won_str  = pytesseract.image_to_string(clean_digits(win_img),  config=SCORE_CFG).strip()
    lose_str = pytesseract.image_to_string(clean_digits(lose_img), config=SCORE_CFG).strip()
    times_text = [[time_str], [won_str], [lose_str]]
    time = extract_time(times_text)
    if time is None:
        return []

    return [(time, msg.strip()) for msg in messages]