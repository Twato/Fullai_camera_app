import cv2
import time
import numpy as np
from picamera2 import Picamera2

# =========================================================
# PI CAMERA MOTION THRESHOLD TEST
# สำหรับเทส CAM1 / CAM2 ที่เป็น Pi Camera ไม่ใช่ USB
# =========================================================

# เลือกกล้อง Pi
# ปกติ CAM1 = 0, CAM2 = 1
PICAM_INDEX = 0

# ใช้ขนาดใกล้เคียงโปรแกรมจริง
CAPTURE_SIZE = (1280, 720)
PREVIEW_SIZE = (640, 360)

# =========================
# MOTION CONFIG
# =========================
MOTION_THRESHOLD = 8_000_000
STABLE_TIME = 1.5
FOCUS_DELAY = 0.5

# กัน noise / เงาเล็ก ๆ
MIN_AREA = 25_000
DIFF_THRESHOLD = 30
BLUR_SIZE = 21

# ถ้า print ถี่ไป เพิ่มค่านี้ได้ เช่น 0.2
PRINT_INTERVAL = 0.1


def make_gray(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)
    return gray


def calc_motion(background_gray, frame_rgb):
    gray = make_gray(frame_rgb)

    diff = cv2.absdiff(background_gray, gray)
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # ลด noise และทำพื้นที่ object ให้ต่อเนื่องขึ้น
    mask = cv2.dilate(mask, None, iterations=2)

    motion_score = int(cv2.sumElems(mask)[0])

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_box = None

    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            max_box = cv2.boundingRect(c)

    return gray, mask, motion_score, int(max_area), max_box


def draw_info(frame_rgb, score, area, status, max_box):
    preview = cv2.resize(frame_rgb, PREVIEW_SIZE)
    preview = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)

    scale_x = PREVIEW_SIZE[0] / CAPTURE_SIZE[0]
    scale_y = PREVIEW_SIZE[1] / CAPTURE_SIZE[1]

    if max_box is not None:
        x, y, w, h = max_box
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(preview, f"SCORE: {score}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(preview, f"AREA : {area}", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(preview, f"THRESHOLD: {MOTION_THRESHOLD}", (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    color = (0, 0, 255) if "DETECT" in status else (255, 255, 255)
    cv2.putText(preview, status, (15, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(preview, "Q=Quit | B=Reset BG | +/- Threshold | A/Z Area", (15, PREVIEW_SIZE[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return preview


def main():
    global MOTION_THRESHOLD, MIN_AREA

    print("==============================================")
    print("PI CAMERA MOTION THRESHOLD TEST")
    print("==============================================")
    print(f"PICAM_INDEX       = {PICAM_INDEX}")
    print(f"CAPTURE_SIZE      = {CAPTURE_SIZE}")
    print(f"MOTION_THRESHOLD  = {MOTION_THRESHOLD}")
    print(f"STABLE_TIME       = {STABLE_TIME}")
    print(f"MIN_AREA          = {MIN_AREA}")
    print("----------------------------------------------")
    print("Key:")
    print("  q = exit")
    print("  b = reset background")
    print("  + = threshold +1,000,000")
    print("  - = threshold -1,000,000")
    print("  a = min area +5,000")
    print("  z = min area -5,000")
    print("----------------------------------------------")

    picam2 = Picamera2(PICAM_INDEX)

    config = picam2.create_preview_configuration(
        main={"size": CAPTURE_SIZE, "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    print("Camera started. Waiting auto exposure...")
    time.sleep(2.0)

    frame = picam2.capture_array()
    background_gray = make_gray(frame)

    print("Background ready")
    print("----------------------------------------------")

    motion_start = None
    detected = False
    last_print = 0

    try:
        while True:
            frame = picam2.capture_array()

            gray, mask, score, area, max_box = calc_motion(background_gray, frame)

            now = time.time()

            if score >= MOTION_THRESHOLD and area >= MIN_AREA:
                if motion_start is None:
                    motion_start = now

                elapsed = now - motion_start
                status = f"DETECTING {elapsed:.1f}/{STABLE_TIME:.1f}s"

                if elapsed >= STABLE_TIME and not detected:
                    detected = True
                    print("==============================================")
                    print("OBJECT DETECTED")
                    print(f"SCORE = {score}")
                    print(f"AREA  = {area}")
                    print(f"THRESHOLD = {MOTION_THRESHOLD}")
                    print(f"MIN_AREA  = {MIN_AREA}")
                    print("==============================================")
                    time.sleep(FOCUS_DELAY)
            else:
                motion_start = None
                detected = False
                status = "WAIT OBJECT"

            if now - last_print >= PRINT_INTERVAL:
                print(f"CAM{PICAM_INDEX + 1} MOTION SCORE: {score} | AREA: {area} | {status}")
                last_print = now

            preview = draw_info(frame, score, area, status, max_box)
            mask_preview = cv2.resize(mask, PREVIEW_SIZE)

            cv2.imshow("Pi Camera Motion Test", preview)
            cv2.imshow("Motion Mask", mask_preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("b"):
                background_gray = gray.copy()
                motion_start = None
                detected = False
                print("----------------------------------------------")
                print("Background reset")
                print("----------------------------------------------")

            elif key == ord("+") or key == ord("="):
                MOTION_THRESHOLD += 1_000_000
                print(f"MOTION_THRESHOLD = {MOTION_THRESHOLD}")

            elif key == ord("-") or key == ord("_"):
                MOTION_THRESHOLD = max(0, MOTION_THRESHOLD - 1_000_000)
                print(f"MOTION_THRESHOLD = {MOTION_THRESHOLD}")

            elif key == ord("a"):
                MIN_AREA += 5_000
                print(f"MIN_AREA = {MIN_AREA}")

            elif key == ord("z"):
                MIN_AREA = max(0, MIN_AREA - 5_000)
                print(f"MIN_AREA = {MIN_AREA}")

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("Camera stopped")


if __name__ == "__main__":
    main()
