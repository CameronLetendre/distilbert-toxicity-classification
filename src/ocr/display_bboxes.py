import cv2


def draw_bboxes(bboxes):
    for box in bboxes():
        cv2.drawContours(box)