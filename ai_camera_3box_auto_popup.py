from picamera2 import Picamera2
from time import sleep, time
from datetime import datetime
from PIL import Image, ImageTk
from ultralytics import YOLO

import os
import math
import threading
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox


# =========================================================
# CONFIG
# =========================================================
SAVE_DIR = "captures"
MODEL_PATH = "/home/toto/rpicam_apps_1.9.0-2_arm64/models/TOC_detection.pt"

# Camera index
PICAM0_INDEX = 0
PICAM1_INDEX = 1
USB_DEVICE = 0

# Pi Camera
PICAM_PREVIEW_SIZE = (640, 480)
PICAM_CAPTURE_SIZE = (1280, 720)

# USB Camera
USB_CAPTURE_WIDTH = 1280
USB_CAPTURE_HEIGHT = 720
USB_CAPTURE_FPS = 15
USB_PREVIEW_SIZE = (480, 270)

# Preview / AI performance
PREVIEW_INTERVAL = 0.5
DETECT_INTERVAL = 0.5
YOLO_IMAGE_SIZE = 416
CONF_THRES = 0.75

# Capture behavior
USB_STABLE_CAPTURE_DELAY = 3.0
PICAM_FOCUS_DELAY = 3.0
REMOVE_DELAY = 3.0

# TOC rule
INNER_PER_OUTER_BOX = 6

# YOLO stable condition
YOLO_STABLE_CENTER_PX = 25
YOLO_STABLE_SIZE_RATIO = 0.15

# Motion detection for Pi cameras
MOTION_THRESHOLD = 50000
STABLE_THRESHOLD = 70000
STABLE_TIME = 1.5

SHOW_DETECTION_BOX = True

os.makedirs(SAVE_DIR, exist_ok=True)


# =========================================================
# GLOBAL
# =========================================================
running = False
current_camera = None
target_count = 0
current_count = 0
last_preview_time = 0

model = None
model_ready = False


# =========================================================
# BASIC UI / UTIL
# =========================================================
def set_status(text):
    try:
        status_label.config(text=text)
        root.update_idletasks()
    except Exception:
        pass
    print(text)


def show_popup(text, bg="#1f8f3a"):
    """
    Popup ตรงกลางหน้าจอ
    ใช้แจ้งหลัง CAM0 + CAM1 ถ่ายครบ 1 Inner Box
    และหลัง CAM3 / TOC ถ่ายครบ 1 Outer Box
    """
    try:
        popup_label.config(
            text=f"[ OK ] {text}",
            bg=bg
        )

        popup_label.place(
            relx=0.5,
            rely=0.55,
            anchor="center"
        )

        root.after(
            3000,
            lambda: popup_label.place_forget()
        )
    except Exception as e:
        print("Popup Error:", e)


def make_folder(name):
    folder = os.path.join(SAVE_DIR, name)
    os.makedirs(folder, exist_ok=True)
    return folder


def filename(camera_name):
    folder = make_folder(camera_name)
    name = datetime.now().strftime(f"{camera_name}_%Y%m%d_%H%M%S.jpg")
    return os.path.join(folder, name)


def update_preview(frame, is_bgr=False):
    global last_preview_time

    now = time()
    if now - last_preview_time < PREVIEW_INTERVAL:
        return

    last_preview_time = now

    try:
        display = cv2.resize(frame, USB_PREVIEW_SIZE)

        if is_bgr:
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(display)
        imgtk = ImageTk.PhotoImage(image=img)

        preview_label.imgtk = imgtk
        preview_label.config(image=imgtk)

    except Exception as e:
        print("Preview Error:", e)


# =========================================================
# MODEL LOAD
# =========================================================
def load_model_once():
    global model, model_ready

    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        set_status("Status: Loading YOLO Model...")

        model = YOLO(MODEL_PATH)

        # warmup model 1 รอบ ให้ตอนถึง TOC ไม่หน่วง
        dummy = np.zeros((YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE, 3), dtype=np.uint8)
        model.predict(
            source=dummy,
            conf=CONF_THRES,
            imgsz=YOLO_IMAGE_SIZE,
            verbose=False
        )

        model_ready = True
        set_status("Status: YOLO Model Ready")
    else:
        model_ready = True
        set_status("Status: YOLO Model Already Ready")


# =========================================================
# MOTION / STABLE FOR PI CAMERA
# =========================================================
def gray_from_frame(frame, is_bgr=False):
    if is_bgr:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    return gray


def motion_score(bg, current):
    diff = cv2.absdiff(bg, current)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return int(np.sum(thresh))


def wait_stable_picam(cam):
    stable_start = None
    prev = None

    while running:
        frame = cam.capture_array()
        update_preview(frame, is_bgr=False)

        gray = gray_from_frame(frame, is_bgr=False)

        if prev is None:
            prev = gray
            sleep(0.2)
            continue

        score = motion_score(prev, gray)
        print("PICAM STABLE SCORE:", score)

        if score < STABLE_THRESHOLD:
            if stable_start is None:
                stable_start = time()

            if time() - stable_start >= STABLE_TIME:
                return gray
        else:
            stable_start = None

        prev = gray
        sleep(0.2)

    return None


def focus_with_preview_picam(cam, camera_name):
    set_status(f"{camera_name}: Auto Focus {PICAM_FOCUS_DELAY:.1f} sec")

    try:
        cam.set_controls({
            "AfMode": 1,
            "AfTrigger": 0
        })
    except Exception as e:
        print("AF control warning:", e)

    start_time = time()

    while running and time() - start_time < PICAM_FOCUS_DELAY:
        frame = cam.capture_array()
        update_preview(frame, is_bgr=False)
        sleep(0.2)

    try:
        cam.set_controls({
            "AfMode": 2,
            "AfTrigger": 0
        })
    except Exception as e:
        print("AF continuous warning:", e)


def run_picamera(camera_index, camera_name):
    global current_camera

    cam = None

    try:
        set_status(f"{camera_name}: Opening Camera")

        cam = Picamera2(camera_index)
        current_camera = cam

        config = cam.create_preview_configuration(
            main={
                "size": PICAM_PREVIEW_SIZE,
                "format": "RGB888"
            }
        )
        cam.configure(config)
        cam.start()
        sleep(3)

        try:
            cam.set_controls({
                "AfMode": 2,
                "AfTrigger": 0
            })
        except Exception as e:
            print("AF setup warning:", e)

        set_status(f"{camera_name}: Loading Background")
        bg = wait_stable_picam(cam)

        if bg is None:
            return

        set_status(f"{camera_name}: Ready / Waiting Object")

        while running:
            frame = cam.capture_array()
            update_preview(frame, is_bgr=False)

            gray = gray_from_frame(frame, is_bgr=False)
            score = motion_score(bg, gray)
            print(f"{camera_name} MOTION SCORE:", score)

            if score > MOTION_THRESHOLD:
                set_status(f"{camera_name}: Object Detected")
                sleep(0.5)

                focus_with_preview_picam(cam, camera_name)

                filepath = filename(camera_name)
                set_status(f"{camera_name}: Capturing Full Image")

                still_config = cam.create_still_configuration(
                    main={"size": PICAM_CAPTURE_SIZE}
                )

                cam.switch_mode_and_capture_file(still_config, filepath)

                print("Saved:", filepath)
                set_status(f"{camera_name}: Saved Image")
                sleep(REMOVE_DELAY)
                break

            sleep(0.3)

    except Exception as e:
        print(f"{camera_name} Error:", e)
        set_status(f"{camera_name}: Error {e}")

    finally:
        try:
            if cam is not None:
                cam.stop()
                cam.close()
        except Exception as e:
            print("Close camera error:", e)

        current_camera = None
        set_status(f"{camera_name}: Closed")
        sleep(1)


def run_picamera_direct_capture(camera_index, camera_name):
    global current_camera

    cam = None

    try:
        set_status(f"{camera_name}: Opening Camera")

        cam = Picamera2(camera_index)
        current_camera = cam

        config = cam.create_preview_configuration(
            main={
                "size": PICAM_PREVIEW_SIZE,
                "format": "RGB888"
            }
        )
        cam.configure(config)
        cam.start()
        sleep(2)

        try:
            cam.set_controls({
                "AfMode": 2,
                "AfTrigger": 0
            })
        except Exception as e:
            print("AF setup warning:", e)

        focus_with_preview_picam(cam, camera_name)

        filepath = filename(camera_name)
        set_status(f"{camera_name}: Capturing Full Image")

        still_config = cam.create_still_configuration(
            main={"size": PICAM_CAPTURE_SIZE}
        )

        cam.switch_mode_and_capture_file(still_config, filepath)

        print("Saved:", filepath)
        set_status(f"{camera_name}: Saved Image")
        sleep(REMOVE_DELAY)

    except Exception as e:
        print(f"{camera_name} Error:", e)
        set_status(f"{camera_name}: Error {e}")

    finally:
        try:
            if cam is not None:
                cam.stop()
                cam.close()
        except Exception as e:
            print("Close camera error:", e)

        current_camera = None
        set_status(f"{camera_name}: Closed")
        sleep(1)


# =========================================================
# YOLO USB CAMERA
# =========================================================
def open_usb_camera(device_index, camera_name):
    cap = cv2.VideoCapture(f"/dev/video{device_index}", cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, USB_CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, USB_CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, USB_CAPTURE_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        set_status(f"{camera_name}: Cannot Open USB Camera /dev/video{device_index}")
        return None

    return cap


def detect_label_yolo(frame):
    detected = False
    display_frame = frame.copy()
    best_box = None
    best_conf = 0.0

    results = model.predict(
        source=frame,
        conf=CONF_THRES,
        imgsz=YOLO_IMAGE_SIZE,
        verbose=False
    )

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])

            if conf > best_conf:
                best_conf = conf

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                name = model.names[cls]

                best_box = (x1, y1, x2, y2, conf, name)
                detected = True

    if best_box is not None and SHOW_DETECTION_BOX:
        x1, y1, x2, y2, conf, name = best_box

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            display_frame,
            f"{name} {conf:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return detected, display_frame, best_box


def is_yolo_box_stable(prev_box, current_box):
    if prev_box is None or current_box is None:
        return False

    px1, py1, px2, py2, _, _ = prev_box
    cx1, cy1, cx2, cy2, _, _ = current_box

    prev_cx = (px1 + px2) / 2
    prev_cy = (py1 + py2) / 2
    curr_cx = (cx1 + cx2) / 2
    curr_cy = (cy1 + cy2) / 2

    center_move = ((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2) ** 0.5

    prev_area = max((px2 - px1) * (py2 - py1), 1)
    curr_area = max((cx2 - cx1) * (cy2 - cy1), 1)

    size_change_ratio = abs(curr_area - prev_area) / prev_area

    print(
        f"YOLO STABLE CHECK | center_move={center_move:.1f}px | "
        f"size_change={size_change_ratio:.2f}"
    )

    return (
        center_move <= YOLO_STABLE_CENTER_PX
        and size_change_ratio <= YOLO_STABLE_SIZE_RATIO
    )


def warmup_usb(cap, camera_name):
    set_status(f"{camera_name}: USB Warm Up")
    start = time()

    while running and time() - start < 1.5:
        ret, frame = cap.read()
        if ret:
            update_preview(frame, is_bgr=True)
        sleep(0.1)


def run_usb_camera_ai(device_index, camera_name):
    cap = None
    last_detect_run = 0
    first_detect_time = None
    last_full_frame = None
    last_display_frame = None
    last_stable_box = None

    try:
        if not model_ready or model is None:
            load_model_once()

        set_status(f"{camera_name}: Opening USB Camera")
        cap = open_usb_camera(device_index, camera_name)

        if cap is None:
            return

        warmup_usb(cap, camera_name)

        set_status(f"{camera_name}: Ready / Waiting Label")

        while running:
            ret, frame = cap.read()

            if not ret:
                set_status(f"{camera_name}: Cannot Read Frame")
                sleep(0.2)
                continue

            last_full_frame = frame.copy()
            now = time()

            if last_display_frame is not None:
                update_preview(last_display_frame, is_bgr=True)
            else:
                update_preview(frame, is_bgr=True)

            if now - last_detect_run >= DETECT_INTERVAL:
                last_detect_run = now

                detected, display_frame, current_box = detect_label_yolo(frame)
                last_display_frame = display_frame

                if detected:
                    stable = is_yolo_box_stable(last_stable_box, current_box)

                    if stable:
                        if first_detect_time is None:
                            first_detect_time = now
                            set_status(f"{camera_name}: Label Stable / Hold Still")

                        hold_time = now - first_detect_time

                        set_status(
                            f"{camera_name}: Label Stable "
                            f"{hold_time:.1f}/{USB_STABLE_CAPTURE_DELAY:.1f} sec"
                        )

                        if hold_time >= USB_STABLE_CAPTURE_DELAY:
                            filepath = filename(camera_name)

                            cv2.imwrite(
                                filepath,
                                last_full_frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 100]
                            )

                            print("Saved:", filepath)
                            set_status(f"{camera_name}: Saved Full Image")
                            sleep(REMOVE_DELAY)
                            break

                    else:
                        first_detect_time = None
                        set_status(f"{camera_name}: Label Moving / Waiting Stable")

                    last_stable_box = current_box

                else:
                    first_detect_time = None
                    last_stable_box = None
                    set_status(f"{camera_name}: Ready / Waiting Label")

            sleep(0.03)

    except Exception as e:
        print(f"{camera_name} Error:", e)
        set_status(f"{camera_name}: Error {e}")

    finally:
        if cap is not None:
            cap.release()

        set_status(f"{camera_name}: Closed")
        sleep(1)


# =========================================================
# SEQUENCE
# =========================================================
def sequence_loop():
    global running, current_count

    current_count = 0

    # STEP 1: CAM0 + CAM1 ถ่าย Inner Box ให้ครบจำนวนก่อน
    while running and current_count < target_count:
        inner_no = current_count + 1

        set_status(f"Status: Inner Box {inner_no}/{target_count} - CAM0")
        progress_label.config(
            text=f"Inner Box: {inner_no} / {target_count}"
        )

        run_picamera(PICAM0_INDEX, "cam0_bag")

        if not running:
            break

        set_status(f"Status: Inner Box {inner_no}/{target_count} - CAM1")
        progress_label.config(
            text=f"Inner Box: {inner_no} / {target_count}"
        )

        run_picamera_direct_capture(PICAM1_INDEX, "cam1_box")

        if not running:
            break

        current_count += 1

        progress_label.config(
            text=f"Inner Box Progress: {current_count} / {target_count}"
        )

        # Popup หลัง CAM0 + CAM1 ถ่ายครบ 1 กล่อง
        show_popup(f"INNER BOX {current_count}/{target_count} COMPLETED")

    if not running:
        set_status("Status: Sequence Stopped")
        return

    # STEP 2: CAM3 / TOC
    toc_total = math.ceil(target_count / INNER_PER_OUTER_BOX)

    set_status(f"Status: Inner Complete / Start TOC {toc_total} Box")
    progress_label.config(
        text=f"Outer Box Total: {toc_total}"
    )
    sleep(1)

    for toc_count in range(1, toc_total + 1):
        if not running:
            break

        set_status(f"Status: Outer Box TOC {toc_count}/{toc_total}")
        progress_label.config(
            text=f"Outer Box TOC: {toc_count} / {toc_total}"
        )

        run_usb_camera_ai(USB_DEVICE, "usb_carton")

        if not running:
            break

        progress_label.config(
            text=f"Outer Box TOC Progress: {toc_count} / {toc_total}"
        )

        show_popup(f"OUTER BOX TOC {toc_count}/{toc_total} COMPLETED")

    if running:
        running = False
        set_status("Status: Finish Job")
        show_popup("ALL CAPTURE COMPLETED")
        root.after(3000, show_page1)
    else:
        set_status("Status: Sequence Stopped")


def start_sequence_thread():
    thread = threading.Thread(target=sequence_loop, daemon=True)
    thread.start()


def stop_sequence():
    global running, current_camera

    running = False

    try:
        if current_camera is not None:
            current_camera.stop()
            current_camera.close()
    except Exception:
        pass

    set_status("Status: Stop Requested")
    root.after(1000, show_page1)


def on_close():
    stop_sequence()
    root.destroy()


# =========================================================
# PAGE CONTROL
# =========================================================
def show_page1():
    global running

    running = False

    popup_label.place_forget()
    page2.pack_forget()
    page1.pack(fill="both", expand=True)

    count_entry.delete(0, tk.END)
    count_entry.focus()

    set_page1_status("Please input quantity 1 - 100")


def show_page2():
    page1.pack_forget()
    page2.pack(fill="both", expand=True)


def set_page1_status(text):
    page1_status.config(text=text)


def start_from_input(event=None):
    global target_count, current_count, running

    value = count_entry.get().strip()

    if not value.isdigit():
        messagebox.showwarning("Warning", "Please input number only")
        return

    count = int(value)

    if count < 1 or count > 100:
        messagebox.showwarning("Warning", "Please input number 1 - 100")
        return

    target_count = count
    current_count = 0
    running = True

    show_page2()

    toc_total = math.ceil(target_count / INNER_PER_OUTER_BOX)

    progress_label.config(
        text=f"Inner Box: 0 / {target_count} | TOC Total: {toc_total}"
    )
    set_status("Status: Preparing AI Model...")

    try:
        load_model_once()
    except Exception as e:
        running = False
        messagebox.showerror("YOLO Error", f"Cannot load YOLO model:\n{e}")
        show_page1()
        return

    set_status("Status: Sequence Started")
    start_sequence_thread()


# =========================================================
# GUI
# =========================================================
root = tk.Tk()
root.title("AI Camera 3-Camera Data Collection")
root.geometry("760x520")
root.resizable(False, False)

# PAGE 1
page1 = tk.Frame(root)

title1 = tk.Label(
    page1,
    text="AI Camera Data Collection",
    font=("Arial", 20, "bold")
)
title1.pack(pady=40)

sub1 = tk.Label(
    page1,
    text="Input Quantity",
    font=("Arial", 16)
)
sub1.pack(pady=10)

count_entry = tk.Entry(
    page1,
    font=("Arial", 28),
    justify="center",
    width=8
)
count_entry.pack(pady=10)

hint1 = tk.Label(
    page1,
    text="Input number 1 - 100 and press Enter",
    font=("Arial", 12),
    fg="gray"
)
hint1.pack(pady=10)

page1_status = tk.Label(
    page1,
    text="Please input quantity 1 - 100",
    font=("Arial", 12)
)
page1_status.pack(pady=10)

count_entry.bind("<Return>", start_from_input)

# PAGE 2
page2 = tk.Frame(root)

title2 = tk.Label(
    page2,
    text="AI Camera 3-Camera Data Collection",
    font=("Arial", 16, "bold")
)
title2.pack(pady=5)

preview_frame = tk.Frame(
    page2,
    bg="black",
    width=640,
    height=300
)
preview_frame.pack(pady=5)
preview_frame.pack_propagate(False)

preview_label = tk.Label(preview_frame, bg="black")
preview_label.pack(fill="both", expand=True)

status_label = tk.Label(
    page2,
    text="Status: Idle",
    font=("Arial", 12)
)
status_label.pack(pady=5)

progress_label = tk.Label(
    page2,
    text="Progress: 0 / 0",
    font=("Arial", 12, "bold")
)
progress_label.pack(pady=5)

button_frame = tk.Frame(page2)
button_frame.pack(pady=5)

btn_stop = tk.Button(
    button_frame,
    text="Stop",
    font=("Arial", 14),
    width=18,
    command=stop_sequence
)
btn_stop.pack(side="left", padx=10)

popup_label = tk.Label(
    page2,
    text="",
    font=("Arial", 24, "bold"),
    bg="#1f8f3a",
    fg="white",
    padx=28,
    pady=12,
    relief="flat"
)
popup_label.place_forget()

root.protocol("WM_DELETE_WINDOW", on_close)

page1.pack(fill="both", expand=True)
count_entry.focus()

root.mainloop()
