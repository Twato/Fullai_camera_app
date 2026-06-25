import cv2
import time

CAMERA_DEVICE = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 15

MOTION_THRESHOLD = 8_000_000
STABLE_TIME = 1.5
FOCUS_DELAY = 0.5

MIN_AREA = 25000
DIFF_THRESHOLD = 30
BLUR_SIZE = 21

cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

if not cap.isOpened():
    print("Cannot open camera")
    raise SystemExit

print("Waiting background...")
time.sleep(2)

ret, bg = cap.read()
if not ret:
    print("Cannot read background")
    raise SystemExit

bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
bg_gray = cv2.GaussianBlur(bg_gray, (BLUR_SIZE, BLUR_SIZE), 0)

motion_start = None
detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)

    diff = cv2.absdiff(bg_gray, gray)
    _, thresh = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    motion_score = int(cv2.sumElems(thresh)[0])

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_box = None
    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            max_box = cv2.boundingRect(c)

    print(f"CAM1 MOTION SCORE: {motion_score} | AREA: {int(max_area)}")

    if motion_score >= MOTION_THRESHOLD and max_area >= MIN_AREA:
        if motion_start is None:
            motion_start = time.time()

        elapsed = time.time() - motion_start
        status = f"DETECTING {elapsed:.1f}/{STABLE_TIME:.1f}s"

        if elapsed >= STABLE_TIME and not detected:
            detected = True
            print("=" * 50)
            print("OBJECT DETECTED")
            print(f"SCORE : {motion_score}")
            print(f"AREA  : {int(max_area)}")
            print("=" * 50)
            time.sleep(FOCUS_DELAY)
    else:
        motion_start = None
        detected = False
        status = "WAIT OBJECT"

    preview = frame.copy()
    if max_box:
        x, y, w, h = max_box
        cv2.rectangle(preview, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.putText(preview, f"SCORE: {motion_score}", (20,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(preview, f"AREA: {int(max_area)}", (20,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(preview, status, (20,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0,0,255) if "DETECT" in status else (255,255,255), 2)

    cv2.imshow("Motion Threshold Test", preview)
    cv2.imshow("Motion Mask", thresh)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        bg_gray = gray.copy()
        motion_start = None
        detected = False
        print("Background Reset")

cap.release()
cv2.destroyAllWindows()
