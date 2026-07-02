#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test CAM1 / CAM2 YOLO detection models from image paths only.
- No camera open
- No preview
- No autofocus
- Supports testing CAM1 image + CAM2 image in one run
- Saves annotated image, ROI crops, and JSON result

Usage examples:
  python3 test_cam12_image_paths_detection.py --cam1-image /path/to/cam1.jpg --cam2-image /path/to/cam2.jpg
  python3 test_cam12_image_paths_detection.py --cam1-image /path/to/cam1.jpg --conf 0.25 --imgsz 1280
  python3 test_cam12_image_paths_detection.py --cam2-image /path/to/cam2.jpg --cam2-model /path/to/model.pt

Default models:
  CAM1 TOAVN : /home/toto/AI_CAMERA_TEST_YOLO_ROI/Models/TOA_model/TOAVN_OBB_V1.pt
  CAM2 TOB   : /home/toto/AI_CAMERA_TEST_YOLO_ROI/Models/TOB_model/TOB_OBB_V1.pt
"""

import os
import cv2
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from ultralytics import YOLO

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# You can change these default paths here.
DEFAULT_CAM1_MODEL = "/home/toto/AI_CAMERA_TEST_YOLO_ROI/Models/TOA_model/TOAVN_OBB_V1.pt"
DEFAULT_CAM2_MODEL = "/home/toto/AI_CAMERA_TEST_YOLO_ROI/Models/TOB_model/TOB_OBB_V1.pt"

OUTPUT_ROOT = os.path.join(APP_DIR, "output_image_detection_test")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def safe_name(value: str) -> str:
    value = str(value or "unknown")
    out = []
    for ch in value:
        if ch.isalnum() or ch in ("_", "-", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "unknown"


def crop_by_xyxy(img_bgr, xyxy, pad: int = 4):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None, [x1, y1, x2, y2]
    return img_bgr[y1:y2, x1:x2].copy(), [x1, y1, x2, y2]


def get_obb_points(result) -> Optional[List[List[List[float]]]]:
    """Return list of OBB corner points if model returns OBB results."""
    try:
        if getattr(result, "obb", None) is None:
            return None
        obb = result.obb
        if obb is None or getattr(obb, "xyxyxyxy", None) is None:
            return None
        pts = obb.xyxyxyxy.detach().cpu().numpy().tolist()
        return pts
    except Exception:
        return None


def parse_detections(result, names: Dict[int, str]) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    if result is None:
        return detections

    obb_points = get_obb_points(result)

    # OBB model case
    if getattr(result, "obb", None) is not None and result.obb is not None:
        obb = result.obb
        try:
            n = len(obb)
        except Exception:
            n = 0
        for i in range(n):
            try:
                if getattr(obb, "xyxy", None) is not None:
                    xyxy = obb.xyxy[i].detach().cpu().numpy().tolist()
                else:
                    pts = obb_points[i]
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    xyxy = [min(xs), min(ys), max(xs), max(ys)]
                conf = float(obb.conf[i].detach().cpu().item())
                cls_id = int(obb.cls[i].detach().cpu().item())
                name = str(names.get(cls_id, cls_id))
                det = {
                    "name": name,
                    "class_id": cls_id,
                    "conf": conf,
                    "box_float": xyxy,
                    "box": [int(round(v)) for v in xyxy],
                    "type": "obb",
                }
                if obb_points is not None:
                    det["obb_points"] = obb_points[i]
                detections.append(det)
            except Exception as e:
                print(f"Parse OBB detection error index={i}: {e}")
        detections.sort(key=lambda d: (d["class_id"], -d["conf"]))
        return detections

    # Normal bbox model case
    if getattr(result, "boxes", None) is None or result.boxes is None:
        return detections

    boxes = result.boxes
    for i in range(len(boxes)):
        try:
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
                "type": "bbox",
            })
        except Exception as e:
            print(f"Parse bbox detection error index={i}: {e}")

    detections.sort(key=lambda d: (d["class_id"], -d["conf"]))
    return detections


def draw_detections(img_bgr, detections: List[Dict[str, Any]]):
    out = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        name = det["name"]
        conf = det["conf"]

        # Draw OBB polygon if available, else rectangle.
        if "obb_points" in det:
            pts = det["obb_points"]
            poly = [(int(round(x)), int(round(y))) for x, y in pts]
            for j in range(len(poly)):
                cv2.line(out, poly[j], poly[(j + 1) % len(poly)], (0, 255, 0), 2)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{name} {conf:.2f}"
        y_text = max(25, y1 - 8)
        cv2.putText(out, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def run_one(role: str, image_path: str, model_path: str, conf: float, imgsz: int, save_crop_pad: int, show: bool) -> Dict[str, Any]:
    role = role.lower()
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    h, w = img_bgr.shape[:2]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_base = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = ensure_dir(os.path.join(OUTPUT_ROOT, f"{ts}_{role}_{safe_name(image_base)}"))
    roi_dir = ensure_dir(os.path.join(out_dir, "roi"))

    print("=" * 88)
    print(f"ROLE        : {role}")
    print(f"MODEL       : {model_path}")
    print(f"IMAGE       : {image_path}")
    print(f"IMAGE SHAPE : {w}x{h}")
    print(f"CONF        : {conf}")
    print(f"IMGSZ       : {imgsz}")
    print("Loading model...")
    model = YOLO(model_path)
    names = model.names
    print(f"MODEL NAMES : {names}")

    print("Running detection...")
    t0 = time.time()
    results = model.predict(img_bgr, imgsz=imgsz, conf=conf, verbose=False)
    duration = time.time() - t0
    result0 = results[0] if results else None
    detections = parse_detections(result0, names)

    annotated = draw_detections(img_bgr, detections)
    annotated_path = os.path.join(out_dir, "annotated.jpg")
    cv2.imwrite(annotated_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

    full_copy_path = os.path.join(out_dir, "full.jpg")
    cv2.imwrite(full_copy_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    saved_detections = []
    per_name_count: Dict[str, int] = {}
    for det in detections:
        crop, fixed_box = crop_by_xyxy(img_bgr, det["box"], pad=save_crop_pad)
        item = dict(det)
        item["box"] = fixed_box
        if crop is not None:
            name = safe_name(det["name"])
            per_name_count[name] = per_name_count.get(name, 0) + 1
            roi_name = f"{name}_{per_name_count[name]}_{det['conf']:.2f}.jpg"
            roi_path = os.path.join(roi_dir, roi_name)
            cv2.imwrite(roi_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            item["roi_path"] = roi_path
            item["roi_shape"] = {"width": int(crop.shape[1]), "height": int(crop.shape[0])}
        saved_detections.append(item)

    result_json = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "role": role,
        "model_path": model_path,
        "model_names": {str(k): v for k, v in names.items()},
        "image_path": image_path,
        "image_shape": {"width": int(w), "height": int(h)},
        "conf": conf,
        "imgsz": imgsz,
        "duration_sec": round(duration, 4),
        "detection_count": len(detections),
        "detections": saved_detections,
        "output_dir": out_dir,
        "full_copy_path": full_copy_path,
        "annotated_path": annotated_path,
        "roi_dir": roi_dir,
    }
    json_path = os.path.join(out_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    print("-" * 88)
    print(f"DETECTION COUNT : {len(detections)}")
    if detections:
        for i, det in enumerate(detections, start=1):
            print(f"{i:02d}. {det['name']:<16} conf={det['conf']:.3f} box={det['box']} type={det.get('type')}")
    else:
        print("No detection found. Try --conf 0.05 or --imgsz 640/1280/2304")
    print("-" * 88)
    print(f"FULL COPY : {full_copy_path}")
    print(f"ANNOTATED : {annotated_path}")
    print(f"ROI DIR   : {roi_dir}")
    print(f"JSON      : {json_path}")
    print(f"TIME      : {duration:.3f}s")
    print("=" * 88)

    if show:
        display_w = 1280
        display_h = int(h * (display_w / max(w, 1)))
        display = cv2.resize(annotated, (display_w, display_h)) if w > display_w else annotated
        cv2.imshow(f"{role} detection", display)
        print("Press any key on image window to continue...")
        cv2.waitKey(0)
        cv2.destroyWindow(f"{role} detection")

    return result_json


def main():
    parser = argparse.ArgumentParser(description="Test CAM1/CAM2 YOLO detection models from image paths only")
    parser.add_argument("--cam1-image", default=None, help="Image path for CAM1 TOAVN model")
    parser.add_argument("--cam2-image", default=None, help="Image path for CAM2 TOB model")
    parser.add_argument("--cam1-model", default=DEFAULT_CAM1_MODEL, help="CAM1 TOAVN model path")
    parser.add_argument("--cam2-model", default=DEFAULT_CAM2_MODEL, help="CAM2 TOB model path")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size")
    parser.add_argument("--pad", type=int, default=4, help="ROI crop padding")
    parser.add_argument("--show", action="store_true", help="Show annotated result window")
    args = parser.parse_args()

    if not args.cam1_image and not args.cam2_image:
        print("Please provide at least one image path:")
        print("  python3 test_cam12_image_paths_detection.py --cam1-image /path/to/cam1.jpg")
        print("  python3 test_cam12_image_paths_detection.py --cam2-image /path/to/cam2.jpg")
        print("  python3 test_cam12_image_paths_detection.py --cam1-image /path/to/cam1.jpg --cam2-image /path/to/cam2.jpg")
        raise SystemExit(2)

    summary = []
    if args.cam1_image:
        summary.append(run_one("cam1", args.cam1_image, args.cam1_model, args.conf, args.imgsz, args.pad, args.show))

    if args.cam2_image:
        summary.append(run_one("cam2", args.cam2_image, args.cam2_model, args.conf, args.imgsz, args.pad, args.show))

    # Save combined summary for this run.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_dir(OUTPUT_ROOT)
    summary_path = os.path.join(OUTPUT_ROOT, f"summary_{ts}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"time": datetime.now().isoformat(timespec="seconds"), "results": summary}, f, ensure_ascii=False, indent=2)
    print(f"SUMMARY JSON: {summary_path}")


if __name__ == "__main__":
    main()
