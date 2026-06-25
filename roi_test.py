import os
import cv2
import numpy as np

from config_test import CENTER_ROI_W, CENTER_ROI_H, OCR_SCALE, ROI_DIR, OCR_DIR

def crop_center(image, roi_w=CENTER_ROI_W, roi_h=CENTER_ROI_H):
    h, w = image.shape[:2]

    x = int((w - roi_w) / 2)
    y = int((h - roi_h) / 2)

    x = max(0, x)
    y = max(0, y)

    x2 = min(w, x + roi_w)
    y2 = min(h, y + roi_h)

    crop = image[y:y2, x:x2]

    return crop, [x, y, x2, y2]

def crop_by_box(image, box, pad=0):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box

    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad)
    y2 = min(h, int(y2) + pad)

    crop = image[y1:y2, x1:x2]

    return crop, [x1, y1, x2, y2]

def preprocess_ocr(crop):
    if crop is None or crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=OCR_SCALE,
        fy=OCR_SCALE,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.convertScaleAbs(gray, alpha=1.25, beta=5)

    sharpen_kernel = np.array([
        [0, -0.5, 0],
        [-0.5, 3, -0.5],
        [0, -0.5, 0]
    ], dtype=np.float32)

    gray = cv2.filter2D(gray, -1, sharpen_kernel)

    return gray

def save_roi_files(image_path, crop, processed, prefix):
    base = os.path.splitext(os.path.basename(image_path))[0]

    roi_path = os.path.join(ROI_DIR, f"{base}_{prefix}_roi.jpg")
    ocr_path = os.path.join(OCR_DIR, f"{base}_{prefix}_ocr.jpg")

    cv2.imwrite(roi_path, crop)

    if processed is not None:
        cv2.imwrite(ocr_path, processed)

    return roi_path, ocr_path
