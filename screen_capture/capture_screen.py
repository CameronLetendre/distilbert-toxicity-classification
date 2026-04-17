from mss import mss
import time

def init_capture():

    while True:
        with mss() as sct:
            sct.shot()

init_capture()


