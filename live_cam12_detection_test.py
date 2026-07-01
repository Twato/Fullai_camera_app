#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live detection test for CAM1 / CAM2 label detection models.
- Opens Raspberry Pi camera by Picamera2
- Runs YOLO detection live
- Shows preview with boxes
- Press S to save current frame + ROI crops
- Press Q / ESC to quit

Usage:
  python3 live_cam12_detection_test.py --cam cam1
  python3 live_cam12_detection_test.py --cam cam2

Optional:
  python3 live_cam12_detection_test.py --cam cam1 --conf 0.05 --imgsz 1280
  python3 live_cam12_detection_test.py --cam cam2 --camera-index 0 --conf 0.05
"""

import os
import cv2
import time
import json
import argparse
from datetime import datetime

from ultralytics import YOLO

try:
    from picamera2 import Picamera2
except Exception as e:
    Picamera2 = None

# =========================
# DEFAULT CONFIG
# =========================
APP_DIR = os.path.dirname(os.path.abspath(__file__))

CAM1_MODEL = "/home/toto/AI_CAMERA_TEST_YOLO_ROI/Models/TOA_model/Models_detection_VNTOA_V1.pt"
CAM2_MODEL = "/home/toto/AI_CAMERA_TEST_YOLO_ROI/Models/TOB_model/Models_detection_TOB_V1.pt"

# Default based on your project config: CAM1=Pi index 1, CAM2=Pi index 0 in recent logs.
CAM1_INDEX = 1
CAM2_INDEX = 0

CAPTURE_W = 2304
CAPTURE_H = 1296
PREVIEW_W = 1280
PREVIEW_H = 720

OUTPUT_ROOT = os.path.join(APP_DIR, "output_live_detection_test")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def crop_by_xyxy(img, xyxy, pad=4):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None, [x1, y1, x2, y2]
    return img[y1:y2, x1:x2].copy(), [x1, y1, x2, y2]


def draw_boxes(img_bgr, detections):
    out = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        name = det["name"]
        conf = det["conf"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{name} {conf:.2f}"
        y_text = max(25, y1 - 8)
        cv2.putText(out, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def parse_detections(result, names):
    detections = []
    if result is None or result.boxes is None:
        return detections
    boxes = result.boxes
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].detach().cpu().numpy().tolist()
        conf = float(boxes.conf[i].detach().cpu().item())
        cls_id = int(boxes.cls[i].detach().cpu().item())
        name = str(names.get(cls_id, cls_id))
        detections.append({
            "name": name,
            "class_id": cls_id,
            "conf": conf,
            "box_float": xyxy,
            "box": [int(round(v)) for v in xyxy],
        })
    detections.sort(key=lambda d: (d["class_id"], -d["conf"]))
    return detections


def save_snapshot(frame_rgb, annotated_bgr, detections, cam_name):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = ensure_dir(os.path.join(OUTPUT_ROOT, f"{ts}_{cam_name}"))
    roi_dir = ensure_dir(os.path.join(out_dir, "roi"))

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    full_path = os.path.join(out_dir, "full.jpg")
    ann_path = os.path.join(out_dir, "annotated.jpg")
    cv2.imwrite(full_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(ann_path, annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    saved_rois = []
    per_name_count = {}
    for det in detections:
        crop, fixed_box = crop_by_xyxy(frame_bgr, det["box"], pad=4)
        if crop is None:
            continue
        name = det["name"]
        per_name_count[name] = per_name_count.get(name, 0) + 1
        roi_name = f"{name}_{per_name_count[name]}_{det['conf']:.2f}.jpg".replace("/", "_")
        roi_path = os.path.join(roi_dir, roi_name)
        cv2.imwrite(roi_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        item = dict(det)
        item["box"] = fixed_box
        item["roi_path"] = roi_path
        saved_rois.append(item)

    result = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "camera": cam_name,
        "full_path": full_path,
        "annotated_path": ann_path,
        "roi_dir": roi_dir,
        "image_shape": {"width": int(frame_rgb.shape[1]), "height": int(frame_rgb.shape[0])},
        "detection_count": len(detections),
        "detections": saved_rois,
    }
    json_path = os.path.join(out_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"SAVED FULL      : {full_path}")
    print(f"SAVED ANNOTATED : {ann_path}")
    print(f"SAVED ROI DIR   : {roi_dir}")
    print(f"SAVED JSON      : {json_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Live Pi camera YOLO detection test for CAM1/CAM2")
    parser.add_argument("--cam", choices=["cam1", "cam2"], required=True, help="Select model/camera role")
    parser.add_argument("--camera-index", type=int, default=None, help="Override Picamera2 camera index")
    parser.add_argument("--model", default=None, help="Override YOLO model path")
    parser.add_argument("--conf", type=float, default=0.10, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size")
    parser.add_argument("--width", type=int, default=CAPTURE_W, help="Camera capture width")
    parser.add_argument("--height", type=int, default=CAPTURE_H, help="Camera capture height")
    parser.add_argument("--every", type=float, default=0.25, help="Detection interval seconds")
    args = parser.parse_args()

    if Picamera2 is None:
        raise RuntimeError("picamera2 is not available. Run this on Raspberry Pi.")

    cam_name = args.cam.lower()
    model_path = args.model or (CAM1_MODEL if cam_name == "cam1" else CAM2_MODEL)
    camera_index = args.camera_index
    if camera_index is None:
        camera_index = CAM1_INDEX if cam_name == "cam1" else CAM2_INDEX

    print("=" * 80)
    print(f"LIVE DETECTION TEST")
    print(f"CAMERA ROLE  : {cam_name}")
    print(f"CAMERA INDEX : {camera_index}")
    print(f"MODEL        : {model_path}")
    print(f"RESOLUTION   : {args.width}x{args.height}")
    print(f"CONF         : {args.conf}")
    print(f"IMGSZ        : {args.imgsz}")
    print("KEYS         : S=save, Q/ESC=quit")
    print("=" * 80)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print("Loading model...")
    model = YOLO(model_path)
    names = model.names
    print(f"MODEL NAMES  : {names}")

    picam2 = Picamera2(camera_num=camera_index)
    config = picam2.create_preview_configuration(
        main={"size": (int(args.width), int(args.height)), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.8)

    last_detect_time = 0.0
    latest_detections = []
    latest_annotated = None
    latest_frame_rgb = None
    fps_t0 = time.time()
    fps_count = 0
    fps = 0.0

    try:
        while True:
            frame_rgb = picam2.capture_array()
            latest_frame_rgb = frame_rgb
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            now = time.time()
            if now - last_detect_time >= float(args.every):
                last_detect_time = now
                results = model.predict(frame_rgb, imgsz=args.imgsz, conf=args.conf, verbose=False)
                latest_detections = parse_detections(results[0] if results else None, names)

                if latest_detections:
                    detail = ", ".join([f"{d['name']}:{d['conf']:.2f}" for d in latest_detections])
                else:
                    detail = "NO DETECTION"
                print(f"DET={len(latest_detections)} | {detail}")

            annotated = draw_boxes(frame_bgr, latest_detections)
            latest_annotated = annotated

            fps_count += 1
            if time.time() - fps_t0 >= 1.0:
                fps = fps_count / (time.time() - fps_t0)
                fps_t0 = time.time()
                fps_count = 0

            display = cv2.resize(annotated, (PREVIEW_W, PREVIEW_H))
            cv2.putText(display, f"{cam_name.upper()} DET={len(latest_detections)} FPS={fps:.1f} CONF={args.conf} IMGSZ={args.imgsz}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Live CAM1/CAM2 Detection Test", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                if latest_frame_rgb is not None and latest_annotated is not None:
                    save_snapshot(latest_frame_rgb, latest_annotated, latest_detections, cam_name)
    finally:
        try:
            picam2.stop()
            picam2.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
