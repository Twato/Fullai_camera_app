import os
import argparse
import json
from datetime import datetime

import cv2
from ultralytics import YOLO

# =====================
# MODEL PATHS
# =====================
CAM1_MODEL_PATH = "/home/toto/AI_CAMERA_TEST_YOLO_ROI/Models/TOA_model/Models_detection_VNTOA_V1.pt"
CAM2_MODEL_PATH = "/home/toto/AI_CAMERA_TEST_YOLO_ROI/Models/TOB_model/Models_detection_TOB_V1.pt"

# Default output folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(BASE_DIR, "output_detection_test")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def crop_by_xyxy(img, xyxy, pad=8):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None, [x1, y1, x2, y2]
    return img[y1:y2, x1:x2].copy(), [x1, y1, x2, y2]


def draw_boxes(img, detections):
    out = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        name = det["name"]
        conf = det["conf"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{name} {conf:.2f}"
        y_text = max(20, y1 - 8)
        cv2.putText(out, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    return out


def run_detection(camera_name, model_path, image_path, conf=0.25, imgsz=1280, pad=8, output_root=OUTPUT_ROOT):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_base = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = ensure_dir(os.path.join(output_root, f"{ts}_{camera_name}_{image_base}"))
    roi_dir = ensure_dir(os.path.join(out_dir, "roi"))

    print("=" * 80)
    print(f"CAMERA      : {camera_name}")
    print(f"MODEL       : {model_path}")
    print(f"IMAGE       : {image_path}")
    print(f"IMAGE SHAPE : {img.shape[1]}x{img.shape[0]}")
    print(f"CONF        : {conf}")
    print(f"IMGSZ       : {imgsz}")
    print("Loading model...")

    model = YOLO(model_path)
    print(f"MODEL NAMES : {model.names}")
    print("Running detection...")

    results = model.predict(source=img, conf=conf, imgsz=imgsz, verbose=False)
    detections = []

    if results and len(results) > 0:
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                det_conf = float(box.conf[0].item())
                name = str(model.names.get(cls_id, cls_id))
                crop, fixed_box = crop_by_xyxy(img, xyxy, pad=pad)
                det = {
                    "index": i + 1,
                    "class_id": cls_id,
                    "name": name,
                    "conf": round(det_conf, 4),
                    "box": fixed_box,
                    "crop_path": "",
                }
                if crop is not None:
                    roi_name = f"{i+1:02d}_{name}_{det_conf:.2f}.jpg".replace("/", "_").replace("\\", "_")
                    roi_path = os.path.join(roi_dir, roi_name)
                    cv2.imwrite(roi_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    det["crop_path"] = roi_path
                detections.append(det)

    annotated = draw_boxes(img, detections)
    annotated_path = os.path.join(out_dir, "annotated.jpg")
    cv2.imwrite(annotated_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

    summary = {
        "camera": camera_name,
        "model_path": model_path,
        "image_path": image_path,
        "image_shape": {"width": img.shape[1], "height": img.shape[0]},
        "conf": conf,
        "imgsz": imgsz,
        "model_names": model.names,
        "detection_count": len(detections),
        "detections": detections,
        "annotated_path": annotated_path,
        "roi_dir": roi_dir,
        "output_dir": out_dir,
    }

    json_path = os.path.join(out_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("-" * 80)
    print(f"DETECTION COUNT : {len(detections)}")
    if detections:
        for det in detections:
            print(f"{det['index']:02d}. {det['name']:<15} conf={det['conf']:<6} box={det['box']} crop={det['crop_path']}")
    else:
        print("No detection found. Try lower --conf 0.10 or different --imgsz 640/1280/2304")
    print("-" * 80)
    print(f"ANNOTATED : {annotated_path}")
    print(f"ROI DIR   : {roi_dir}")
    print(f"JSON      : {json_path}")
    print("=" * 80)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Test CAM1/CAM2 YOLO detection models from image path, no camera capture.")
    parser.add_argument("--cam", choices=["cam1", "cam2", "both"], required=True, help="Which model to test")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence, default 0.25")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO image size, default 1280")
    parser.add_argument("--pad", type=int, default=8, help="Crop padding pixels, default 8")
    parser.add_argument("--out", default=OUTPUT_ROOT, help="Output folder")
    args = parser.parse_args()

    if args.cam in ("cam1", "both"):
        run_detection("cam1", CAM1_MODEL_PATH, args.image, args.conf, args.imgsz, args.pad, args.out)

    if args.cam in ("cam2", "both"):
        run_detection("cam2", CAM2_MODEL_PATH, args.image, args.conf, args.imgsz, args.pad, args.out)


if __name__ == "__main__":
    main()
