import cv2
import numpy as np
from PIL import Image

def estimate_severity(image: Image.Image) -> float:
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green1 = np.array([25,40,40])
    green2 = np.array([95,255,255])
    mask_green = cv2.inRange(hsv, green1, green2)
    yellow1 = np.array([10,30,30])
    yellow2 = np.array([35,255,255])
    brown1 = np.array([5,40,20])
    brown2 = np.array([30,255,180])
    mask_yellow = cv2.inRange(hsv, yellow1, yellow2)
    mask_brown = cv2.inRange(hsv, brown1, brown2)
    mask_bad = cv2.bitwise_or(mask_yellow, mask_brown)
    leaf_mask = cv2.morphologyEx(cv2.bitwise_or(mask_green, mask_bad), cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)))
    bad_inside = cv2.bitwise_and(mask_bad, leaf_mask)
    leaf_area = int(np.count_nonzero(leaf_mask))
    bad_area = int(np.count_nonzero(bad_inside))
    if leaf_area == 0:
        return 0.0
    return 100.0 * bad_area / leaf_area
