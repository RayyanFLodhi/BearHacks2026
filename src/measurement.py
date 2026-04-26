import cv2
import numpy as np
from skimage.morphology import skeletonize


def contour_length_px(contour) -> float:
    return float(cv2.arcLength(contour, closed=False))


def bbox_length_px(bbox) -> int:
    _, _, w, h = bbox
    return int(max(w, h))


def mask_area_px(mask) -> int:
    return int(np.count_nonzero(mask > 0))


def skeleton_length_px(mask) -> int:
    binary = (mask > 0).astype(np.uint8)
    skeleton = skeletonize(binary).astype(np.uint8)
    return int(np.count_nonzero(skeleton))


# TODO: Add pixel-to-cm calibration using marker or camera geometry metadata.
