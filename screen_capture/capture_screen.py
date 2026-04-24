import mss
import mss.tools
from time import sleep
import easyocr
import cv2

VALORANT = {
"time": {"top": 20, "left": 786, "width": 370, "height": 70},
"chat":{"top": 790, "left": 30, "width": 440, "height": 250}
}

def init_capture(game):
    with mss.mss() as sct:
        while True:
            img = sct.grab(game["chat"])
            mss.tools.to_png(img.rgb, img.size, output="test_chat.png")

            img = sct.grab(game["time"])
            mss.tools.to_png(img.rgb, img.size, output="test_time.png")
            sleep(10)


def extract_OCR():
    def clean_image(img):
        img = cv2.imread(img)
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img
    
    def union(boxes):
        xs = [pt[0] for b in boxes for pt in b]
        ys = [pt[1] for b in boxes for pt in b]

        return (min(xs), min(ys), max(xs), max(ys))

    img = clean_image("test_chat2.png")
    reader = easyocr.Reader(['en'])
    results = reader.readtext(img, detail = 1)


    messages = []
    curr = []
    curr_bbox = []


    getting_message = False
    for bbox, text, conf in results:
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



init_capture(VALORANT)